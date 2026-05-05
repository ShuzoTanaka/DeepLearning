# -*- coding: utf-8 -*-
"""
Evaluation & test script for multi-class segmentation
(background / nerve / spinal)
"""

import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import albumentations as albu
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import Fscore


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["background", "nerve", "spinal"]
ACTIVATION = "softmax2d"

TEST_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\data_split\test"
BEST_MODEL_PATH = (
    r"C:\Users\orilab\Desktop\masumoto\smp\checkpoints\20251202_1750_att_unet2d.pth"
)

OUTPUT_ROOT = r"C:\Users\orilab\Desktop\masumoto\smp\output"
NER_PR_DIR = os.path.join(OUTPUT_ROOT, "nerve_pr")
NER_GT_DIR = os.path.join(OUTPUT_ROOT, "nerve_gt")
SPN_PR_DIR = os.path.join(OUTPUT_ROOT, "spinal_pr")
SPN_GT_DIR = os.path.join(OUTPUT_ROOT, "spinal_gt")
RESULT_IMG_DIR = os.path.join(OUTPUT_ROOT, "test_result")


# =========================
# Utility
# =========================
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# =========================
# Dataset
# =========================
class Dataset(BaseDataset):
    CLASSES = ["background", "nerve", "spinal"]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.class_values = [0, 127, 255]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        masks = np.array([(mask == v) for v in self.class_values])
        mask = np.stack(masks, axis=-1).astype("float")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


# =========================
# Loss
# =========================
class MultiClassDiceLoss(nn.Module):
    def __init__(self, class_weights=None, eps=1e-7):
        super().__init__()
        self.class_weights = class_weights
        self.eps = eps
        self.__name__ = "MultiClassDiceLoss"

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = target.float()

        dims = (0, 2, 3)
        intersection = torch.sum(pred * target, dims)
        cardinality = torch.sum(pred + target, dims)
        dice_loss = 1 - (2.0 * intersection + self.eps) / (cardinality + self.eps)

        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights

        return dice_loss.mean()


# =========================
# Dice coefficient on PNGs
# =========================
def dice_coefficient(image_path, predict_path):
    image = cv2.imread(image_path)
    predict = cv2.imread(predict_path)

    prediction_label = predict == 255
    target_label = image == 255

    intersection = np.logical_and(prediction_label, target_label)
    tp = np.sum(intersection)
    fp = np.sum(prediction_label) - tp
    fn = np.sum(target_label) - tp
    tn = np.sum((prediction_label == 0) & (target_label == 0))

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return accuracy, recall, dice, precision


# =========================
# Main
# =========================
def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Prepare output dirs
    for d in [
        OUTPUT_ROOT,
        NER_PR_DIR,
        NER_GT_DIR,
        SPN_PR_DIR,
        SPN_GT_DIR,
        RESULT_IMG_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # Load best model
    best_model = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    best_model = best_model.to(DEVICE)
    best_model.eval()

    # Preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Datasets
    test_dataset = Dataset(
        os.path.join(TEST_DIR, "image"),
        os.path.join(TEST_DIR, "mask"),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)

    # Dataset for visualization (no preprocessing)
    test_dataset_vis = Dataset(
        os.path.join(TEST_DIR, "image"),
        os.path.join(TEST_DIR, "mask"),
        classes=CLASSES,
    )

    # Loss & metrics for smp ValidEpoch
    weights = torch.tensor([0.1, 1.5, 0.5], device=DEVICE)
    loss = MultiClassDiceLoss(class_weights=weights)
    metrics = [Fscore(threshold=0.5)]

    # smp ValidEpoch
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_loader)
    print("Test MultiClassDiceLoss:", logs["MultiClassDiceLoss"])
    print("Test F-score:", logs["fscore"])

    # ---- Simple single-sample visualization (optional) ----
    n_vis = min(55, len(test_dataset) - 1)
    image_vis = test_dataset_vis[n_vis][0].astype("uint8")
    image, gt_mask = test_dataset[n_vis]

    gt_mask = gt_mask.squeeze().transpose(1, 2, 0)

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        pr_mask = best_model(x_tensor)
        pr_mask = torch.softmax(pr_mask, dim=1)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round()).transpose(1, 2, 0)

    visualize(
        image=image_vis,
        bg_gt_mask=gt_mask[..., 0].squeeze(),
        nerve_gt_mask=gt_mask[..., 1].squeeze(),
        spinal_gt_mask=gt_mask[..., 2].squeeze(),
        bg_pr_mask=pr_mask[..., 0].squeeze(),
        nerve_pr_mask=pr_mask[..., 1].squeeze(),
        spinal_pr_mask=pr_mask[..., 2].squeeze(),
    )

    # ---- Detailed per-image Dice (nerve / spinal) ----
    threshold = 0.7
    spinal_bad_nerve_dice = 0.0
    spinal_bad_nerve_num = 0
    spinal_good_nerve_dice = 0.0
    spinal_good_nerve_num = 0

    spinal_list = []
    spinal_good_list = []
    spinal_bad_list = []

    dice_sum = 0.0
    count = 0

    num_samples = len(test_dataset)
    print("Number of test samples:", num_samples)

    with torch.no_grad():
        for i in range(num_samples):
            count += 1
            image_vis = test_dataset_vis[i][0].astype("uint8")
            image, mask = test_dataset[i]

            gt_mask = mask.squeeze().transpose(1, 2, 0)

            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = best_model(x_tensor)  # [1, C, H, W]
            pr_mask = torch.softmax(pr_mask, dim=1)

            # Save multi-channel result image
            pr_np_full = pr_mask.squeeze(0).permute(1, 2, 0).cpu().numpy().copy()
            img_pil = Image.fromarray((pr_np_full * 255).astype(np.uint8))
            img_pil = img_pil.convert("P")
            img_pil.save(os.path.join(RESULT_IMG_DIR, f"test{i+1}_result.png"))

            # For dice of nerve / spinal
            pr_mask_np = (pr_mask.squeeze().cpu().numpy().round()).transpose(1, 2, 0)

            nerve_gt = gt_mask[..., 1]
            spinal_gt = gt_mask[..., 2]
            nerve_np = pr_mask_np[..., 1]
            spinal_np = pr_mask_np[..., 2]

            nerve_np[nerve_np != 0] = 255
            nerve_gt[nerve_gt != 0] = 255
            spinal_np[spinal_np != 0] = 255
            spinal_gt[spinal_gt != 0] = 255

            pil_nerve_pr = Image.fromarray(nerve_np.astype(np.uint8))
            pil_nerve_gt = Image.fromarray(nerve_gt.astype(np.uint8))
            pil_spinal_pr = Image.fromarray(spinal_np.astype(np.uint8))
            pil_spinal_gt = Image.fromarray(spinal_gt.astype(np.uint8))

            nerve_pr_path = os.path.join(NER_PR_DIR, f"nerve{i+1}.png")
            nerve_gt_path = os.path.join(NER_GT_DIR, f"nerve{i+1}.png")
            spinal_pr_path = os.path.join(SPN_PR_DIR, f"spinal{i+1}.png")
            spinal_gt_path = os.path.join(SPN_GT_DIR, f"spinal{i+1}.png")

            pil_nerve_pr.convert("L").save(nerve_pr_path)
            pil_nerve_gt.convert("L").save(nerve_gt_path)
            pil_spinal_pr.convert("L").save(spinal_pr_path)
            pil_spinal_gt.convert("L").save(spinal_gt_path)

            dice_nerve = dice_coefficient(nerve_pr_path, nerve_gt_path)[2]
            dice_spinal = dice_coefficient(spinal_pr_path, spinal_gt_path)[2]

            dice_sum += dice_nerve
            spinal_list.append(dice_spinal)

            if dice_nerve > threshold:
                spinal_good_list.append(dice_nerve)
                spinal_good_nerve_dice += dice_nerve
                spinal_good_nerve_num += 1
            elif dice_nerve < threshold:
                spinal_bad_list.append(dice_nerve)
                spinal_bad_nerve_dice += dice_nerve
                spinal_bad_nerve_num += 1

            if i <= 10:
                visualize(
                    image=image_vis,
                    nerve_gt_mask=gt_mask[..., 1].squeeze(),
                    nerve_pr_mask=pr_mask_np[..., 1].squeeze(),
                )

            print(
                f"[{i+1}/{num_samples}] nerve_dice: {dice_nerve:.4f}, spinal_dice: {dice_spinal:.4f}"
            )

    print("nerve_dice (mean):", dice_sum / count if count > 0 else 0.0)
    if spinal_good_nerve_num > 0:
        print("spinal_good (mean):", spinal_good_nerve_dice / spinal_good_nerve_num)
    if spinal_bad_nerve_num > 0:
        print("spinal_bad (mean):", spinal_bad_nerve_dice / spinal_bad_nerve_num)


if __name__ == "__main__":
    main()
