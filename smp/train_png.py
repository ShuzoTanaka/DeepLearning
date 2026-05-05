# -*- coding: utf-8 -*-
"""
Training script for multi-class segmentation (background / nerve / spinal)
"""

import os
import glob
import traceback
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
DECODER = "unet"

TRAIN_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\data_split\train"
VAL_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\data_split\val"
SAVE_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\checkpoints"

NUM_EPOCHS = 200
BATCH_SIZE = 8
LR_INIT = 1e-3
LR_AFTER_20 = 1e-4


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
    IMAGE_SIZE = 256
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        albu.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            always_apply=True,
            border_mode=0,
        ),
        albu.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


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

        # Mask class values
        self.class_values = [0, 127, 255]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # read mask (grayscale)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        masks = np.array([(mask == v) for v in self.class_values])
        mask = np.stack(masks, axis=-1).astype("float")

        # augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # preprocessing
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
        # For smp.utils logging
        self.__name__ = "MultiClassDiceLoss"

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits → softmax
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
# Main
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Preprocessing function from encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Datasets
    train_dataset = Dataset(
        os.path.join(TRAIN_DIR, "image"),
        os.path.join(TRAIN_DIR, "mask"),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        # augmentation=get_training_augmentation(),  # 必要なら有効化
    )

    valid_dataset = Dataset(
        os.path.join(VAL_DIR, "image"),
        os.path.join(VAL_DIR, "mask"),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
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

    # Epoch runners
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

    # Logging
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

        # log
        x_epoch_data.append(epoch)
        train_dice_loss.append(train_logs["MultiClassDiceLoss"])
        train_f_score.append(train_logs["fscore"])
        valid_dice_loss.append(val_logs["MultiClassDiceLoss"])
        valid_f_score.append(val_logs["fscore"])

        # save best model
        if max_score < val_logs["fscore"]:
            max_score = val_logs["fscore"]
            filename = f"{today_str}_{DECODER}_{ENCODER}.pth"
            save_path = os.path.join(SAVE_DIR, filename)
            torch.save(model, save_path)
            print(f"Model saved: {save_path}")

        # LR scheduling
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
