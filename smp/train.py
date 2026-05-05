# -*- coding: utf-8 -*-
"""
Training script for multi-class segmentation (background / nerve / spinal)
using 2D Axial slices from NIfTI volumes (imagesTr / labelsTr).

imagesTr:
    case001_0000.nii.gz
    case001_0001.nii.gz   ← 今は 0000 だけ使用
labelsTr:
    case001.nii.gz
"""

import os
import glob
import traceback
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib  # NIfTI

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as BaseDataset

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
DECODER = "unet"

DATA_ROOT = r"C:\Users\orilab\Desktop\masumoto\smp\Dataset001_lumber"
IMAGES_TR_DIR = os.path.join(DATA_ROOT, "imagesTr")
LABELS_TR_DIR = os.path.join(DATA_ROOT, "labelsTr")

SAVE_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\checkpoints"

NUM_EPOCHS = 200
BATCH_SIZE = 8
LR_INIT = 1e-3
LR_AFTER_20 = 1e-4

IMAGE_SIZE = 256
VAL_RATIO = 0.2
RANDOM_SEED = 42


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


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
        albu.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
        albu.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, always_apply=True),
        # ノイズ系
        albu.GaussNoise(p=0.2),
        # 幾何
        albu.Perspective(p=0.5),
        # 明るさ・コントラスト
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                # ★ RandomBrightness → RandomBrightnessContrast に統合
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        # シャープ・ぼかし
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        # 色相・彩度など
        albu.OneOf(
            [
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


# =========================
# NIfTI 2D Slice Dataset
# =========================
class NiftiSliceDataset(BaseDataset):
    """
    samples: List[ (image_path, label_path, slice_index) ]
    ラベル値は 0,1,2（background, nerve, spinal）を想定
    """

    def __init__(
        self,
        samples,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.samples = samples
        self.class_values = [0, 1, 2]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, label_path, z = self.samples[i]

        # NIfTI load
        img_vol = nib.load(img_path).get_fdata()
        msk_vol = nib.load(label_path).get_fdata()

        if img_vol.ndim == 4:
            img_vol = img_vol[..., 0]
        if msk_vol.ndim == 4:
            msk_vol = msk_vol[..., 0]

        img_slice = img_vol[:, :, z].astype(np.float32)
        mask_slice = msk_vol[:, :, z].astype(np.int64)  # 0/1/2

        # 画像を 0-255 に rescale して3ch化
        vmin, vmax = img_slice.min(), img_slice.max()
        if vmax > vmin:
            img_norm = (img_slice - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img_slice)
        img_u8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)
        image = np.stack([img_u8, img_u8, img_u8], axis=-1)  # (H, W, 3)

        # mask → one-hot (H, W, 3)
        masks = np.stack(
            [(mask_slice == v) for v in self.class_values], axis=-1
        ).astype("float32")

        # augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=masks)
            image, masks = sample["image"], sample["mask"]

        # preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=masks)
            image, masks = sample["image"], sample["mask"]

        return image, masks


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
# Build slice-level samples
# =========================
def strip_nii_ext(fname: str) -> str:
    """
    'case001_0000.nii.gz' → 'case001_0000'
    'case001.nii'         → 'case001'
    """
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return os.path.splitext(fname)[0]


def build_slice_samples(images_dir, labels_dir):
    """
    imagesTr: case001_0000.nii.gz, case001_0001.nii.gz, ...
    labelsTr: case001.nii.gz

    今回は *_0000.nii.gz のみを使って、
    画像 'case001_0000.nii.gz' → ラベル 'case001.nii.gz' に対応付ける。
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    samples_per_case = {}

    for img_path in image_files:
        base = os.path.basename(img_path)  # e.g. "case001_0000.nii.gz"

        # *_0000.nii.gz だけ使う
        stem = strip_nii_ext(base)  # "case001_0000"
        if stem.endswith("_0000"):
            case_id = stem[:-5]  # "case001"
        else:
            # *_0001.nii.gz などはスキップ（必要になったら後で使う）
            print(f"[INFO] skip image (not _0000): {base}")
            continue

        # label は "case001.nii.gz" or "case001.nii" を探す
        label_nii_gz = os.path.join(labels_dir, case_id + ".nii.gz")
        label_nii = os.path.join(labels_dir, case_id + ".nii")

        if os.path.exists(label_nii_gz):
            label_path = label_nii_gz
        elif os.path.exists(label_nii):
            label_path = label_nii
        else:
            print(f"[WARN] label not found for {base} (case_id={case_id}), skip.")
            continue

        # ここで1症例分の volume shape を見る
        img_vol = nib.load(img_path).get_fdata()
        msk_vol = nib.load(label_path).get_fdata()

        if img_vol.ndim == 4:
            img_vol = img_vol[..., 0]
        if msk_vol.ndim == 4:
            msk_vol = msk_vol[..., 0]

        assert (
            img_vol.shape == msk_vol.shape
        ), f"Shape mismatch: {img_vol.shape} vs {msk_vol.shape} ({base})"

        H, W, Z = img_vol.shape
        print(
            f"[{case_id}] use image: {base}, label: {os.path.basename(label_path)}, shape {H}x{W}x{Z}"
        )

        case_samples = []
        for z in range(Z):
            case_samples.append((img_path, label_path, z))

        samples_per_case[case_id] = case_samples

    return samples_per_case


def split_train_val(samples_per_case, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    case_ids = sorted(samples_per_case.keys())
    rng.shuffle(case_ids)

    n_cases = len(case_ids)
    if n_cases == 0:
        print("[ERROR] No cases found. Check imagesTr / labelsTr path and filenames.")
        return [], [], [], []

    n_val = max(1, int(n_cases * val_ratio))

    val_case_ids = case_ids[:n_val]
    train_case_ids = case_ids[n_val:]

    train_samples = []
    val_samples = []

    for cid in train_case_ids:
        train_samples.extend(samples_per_case[cid])

    for cid in val_case_ids:
        val_samples.extend(samples_per_case[cid])

    print(
        f"#cases total: {n_cases}, train: {len(train_case_ids)}, val: {len(val_case_ids)}"
    )
    print(f"#slices train: {len(train_samples)}, val: {len(val_samples)}")

    return train_samples, val_samples, train_case_ids, val_case_ids


# =========================
# Main
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # ---- NIfTI → slice samples ----
    samples_per_case = build_slice_samples(IMAGES_TR_DIR, LABELS_TR_DIR)
    train_samples, val_samples, train_case_ids, val_case_ids = split_train_val(
        samples_per_case, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )

    if len(train_samples) == 0 or len(val_samples) == 0:
        print("[ERROR] train_samples or val_samples is empty. Stop.")
        return

    print("Train cases:", train_case_ids)
    print("Val   cases:", val_case_ids)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Datasets
    train_dataset = NiftiSliceDataset(
        train_samples,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = NiftiSliceDataset(
        val_samples,
        classes=CLASSES,
        augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    model = model.to(DEVICE)

    # Loss & optimizer & metrics
    weights = torch.tensor([1.0, 1.0, 1.0], device=DEVICE)
    loss = MultiClassDiceLoss(class_weights=weights)

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=LR_INIT),
        ]
    )

    metrics = [
        Fscore(threshold=0.5),
    ]

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    today_str = datetime.now().strftime("%Y%m%d_%H%M")
    max_score = 0.0

    x_epoch_data = []
    train_dice_loss = []
    train_f_score = []
    valid_dice_loss = []
    valid_f_score = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch}")
        try:
            train_logs = train_epoch.run(train_loader)
            val_logs = valid_epoch.run(valid_loader)
        except Exception:
            print("例外が発生しました:")
            traceback.print_exc()
            continue

        x_epoch_data.append(epoch)
        train_dice_loss.append(train_logs["MultiClassDiceLoss"])
        train_f_score.append(train_logs["fscore"])
        valid_dice_loss.append(val_logs["MultiClassDiceLoss"])
        valid_f_score.append(val_logs["fscore"])

        if max_score < val_logs["fscore"]:
            max_score = val_logs["fscore"]
            filename = f"{today_str}_{DECODER}_{ENCODER}_nifti2d.pth"
            save_path = os.path.join(SAVE_DIR, filename)
            torch.save(model, save_path)
            print(f"Model saved: {save_path}")

        if epoch == 20:
            optimizer.param_groups[0]["lr"] = LR_AFTER_20
            print("Decrease learning rate to 1e-4!")

    # Plot curves
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_epoch_data, train_dice_loss, label="train")
    ax1.plot(x_epoch_data, valid_dice_loss, label="validation")
    ax1.set_title("Dice loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x_epoch_data, train_f_score, label="train")
    ax2.plot(x_epoch_data, valid_f_score, label="validation")
    ax2.set_title("F-score")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("fscore")
    ax2.legend(loc="upper left")

    plt.show()


if __name__ == "__main__":
    main()
