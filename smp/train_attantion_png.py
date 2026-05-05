# -*- coding: utf-8 -*-
"""
Training script for multi-class segmentation (background / nerve / spinal)
with Attention U-Net (2D)
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

CLASSES = ["background", "nerve", "spinal"]

TRAIN_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\data_split\train"
VAL_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\data_split\val"
SAVE_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\checkpoints"

NUM_EPOCHS = 200
BATCH_SIZE = 8
LR_INIT = 1e-3
LR_AFTER_20 = 1e-4
IMAGE_SIZE = 256


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


def get_preprocessing():
    """
    シンプルに 0-1 正規化のみ。
    （元の smp の encoder 前処理は使わず、自作 U-Net 前提にする）
    """
    _transform = [
        albu.Lambda(image=lambda x, **k: x / 255.0),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_training_augmentation():
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

        # マスクの値: 背景=0, 神経=127, 硬膜管=255
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
        mask = np.stack(masks, axis=-1).astype("float32")

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
# Attention U-Net (2D)
# =========================
class AttentionBlock2D(nn.Module):
    """
    Attention Gate for 2D U-Net.
    x: encoder feature (skip)
    g: decoder feature (gating)
    """

    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: skip (B, F_x, H, W)
        # g: gating from decoder (B, F_g, H', W')
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # サイズが違う場合は、skip のサイズに合わせる
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # (B, F_x, H, W) * (B, 1, H, W)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionUNet2D(nn.Module):
    """
    Classic 2D U-Net + Attention Gate on skip connections.
    入力: RGB 画像 (B, 3, H, W)
    出力: ログits (B, num_classes, H, W)
    """

    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.center = ConvBlock(512, 1024)

        # Attention blocks (skip connections)
        # center (1024ch) → up4 → d4 は 512ch
        # e4 も 512ch なので F_g=512, F_x=512 が正しい
        self.att4 = AttentionBlock2D(F_g=512, F_x=512, F_int=256)

        # d4 は 512ch → up3 → d3 は 256ch
        # e3 も 256ch
        self.att3 = AttentionBlock2D(F_g=256, F_x=256, F_int=128)

        # d3 は 256ch → up2 → d2 は 128ch
        # e2 も 128ch
        self.att2 = AttentionBlock2D(F_g=128, F_x=128, F_int=64)

        # d2 は 128ch → up1 → d1 は 64ch
        # e1 も 64ch
        self.att1 = AttentionBlock2D(F_g=64, F_x=64, F_int=32)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Segmentation head
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)  # (B, 64, 256, 256)
        p1 = self.pool1(e1)  # (B, 64, 128, 128)

        e2 = self.enc2(p1)  # (B, 128, 128, 128)
        p2 = self.pool2(e2)  # (B, 128, 64, 64)

        e3 = self.enc3(p2)  # (B, 256, 64, 64)
        p3 = self.pool3(e3)  # (B, 256, 32, 32)

        e4 = self.enc4(p3)  # (B, 512, 32, 32)
        p4 = self.pool4(e4)  # (B, 512, 16, 16)

        # ----- Bottleneck -----
        center = self.center(p4)  # (B, 1024, 16, 16)

        # ----- Decoder with Attention -----
        d4 = self.up4(center)  # (B, 512, 32, 32)
        e4_att = self.att4(e4, d4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)  # (B, 512, 32, 32)

        d3 = self.up3(d4)  # (B, 256, 64, 64)
        e3_att = self.att3(e3, d3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)  # (B, 256, 64, 64)

        d2 = self.up2(d3)  # (B, 128, 128, 128)
        e2_att = self.att2(e2, d2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)  # (B, 128, 128, 128)

        d1 = self.up1(d2)  # (B, 64, 256, 256)
        e1_att = self.att1(e1, d1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)  # (B, 64, 256, 256)

        logits = self.seg_head(d1)  # (B, num_classes, 256, 256)
        return logits


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
        """
        pred: (B, C, H, W) logits
        target: (B, C, H, W) one-hot mask
        """
        # logits → softmax で確率化
        pred = F.softmax(pred, dim=1)
        target = target.float()

        dims = (0, 2, 3)
        intersection = torch.sum(pred * target, dims)
        cardinality = torch.sum(pred + target, dims)
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)

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

    # Preprocessing
    preprocessing = get_preprocessing()

    # Dataset
    train_dataset = Dataset(
        os.path.join(TRAIN_DIR, "image"),
        os.path.join(TRAIN_DIR, "mask"),
        preprocessing=preprocessing,
        classes=CLASSES,
        # augmentation=get_training_augmentation(),  # 必要なら有効化
    )

    valid_dataset = Dataset(
        os.path.join(VAL_DIR, "image"),
        os.path.join(VAL_DIR, "mask"),
        preprocessing=preprocessing,
        classes=CLASSES,
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ===== Model: Attention U-Net =====
    model = AttentionUNet2D(in_channels=3, num_classes=len(CLASSES))
    model = model.to(DEVICE)

    # Loss & optimizer & metrics
    weights = torch.tensor([0.1, 1.5, 0.5], device=DEVICE)
    loss = MultiClassDiceLoss(class_weights=weights)

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=LR_INIT),
        ]
    )

    # NOTE: Fscore は smp の util をそのまま使用
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
            filename = f"{today_str}_att_unet2d.pth"
            save_path = os.path.join(SAVE_DIR, filename)
            torch.save(model.state_dict(), save_path)
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
