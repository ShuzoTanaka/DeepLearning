# -*- coding: utf-8 -*-
"""
train.py
PNG画像（image, mask）を使った U-Net (background / nerve / spinal) の学習スクリプト
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as albu
import segmentation_models_pytorch as smp
import os
from datetime import datetime


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["background", "nerve", "spinal"]
ACTIVATION = None
DECODER = "unet"

TRAIN_DIR = r"C:\Users\orilab\Desktop\masumoto\2dunet\data\train"
VAL_DIR = r"C:\Users\orilab\Desktop\masumoto\2dunet\data\val"

# ★ 出力ディレクトリ & 日付付きファイル名
OUTPUT_DIR = r"C:\Users\orilab\Desktop\masumoto\2dunet\output"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 例: 20251208_152310
SAVE_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}_2dunet.pth")

NUM_EPOCHS = 40
BATCH_SIZE = 6
LR_INIT = 1e-3
LR_AFTER_20 = 1e-4
IMAGE_SIZE = 256
EPS = 1e-7


# =========================
# Utility
# =========================
def visualize(**images):
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


class FscoreMetric:
    """
    multi-class 用の F1 (macro) 指標
    y_pred: (B, C, H, W) ロジット
    y_true: (B, C, H, W) one-hot
    """

    def __init__(self, ignore_background=True, eps=1e-7):
        self.ignore_background = ignore_background
        self.eps = eps
        self.__name__ = "fscore"

    def __call__(self, y_pred, y_true):
        # 予測クラス (0,1,2)
        y_pred_cls = y_pred.argmax(dim=1)  # (B,H,W)
        # 正解クラス (0,1,2)
        y_true_cls = y_true.argmax(dim=1)  # (B,H,W)

        num_classes = y_pred.shape[1]

        f_per_class = []
        start_class = 1 if self.ignore_background else 0  # 背景(0)は無視するかどうか

        for c in range(start_class, num_classes):
            pred_c = (y_pred_cls == c).float()
            true_c = (y_true_cls == c).float()

            tp = (pred_c * true_c).sum()
            fp = (pred_c * (1 - true_c)).sum()
            fn = ((1 - pred_c) * true_c).sum()

            f = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
            f_per_class.append(f)

        if len(f_per_class) == 0:
            return torch.tensor(0.0, device=y_pred.device)
        return torch.stack(f_per_class).mean()


class MultiClassDiceLoss(nn.Module):
    """
    y_pred: (B, C, H, W) ロジット
    y_true: one-hot (B, C, H, W)
    """

    def __init__(self, class_weights=None, eps=1e-7):
        super().__init__()
        self.eps = eps
        # Python の list / np.array のまま持っておく（デバイスは forward で揃える）
        self.class_weights = class_weights
        self.__name__ = "dice_loss"

    def forward(self, y_pred, y_true):
        # softmax → 確率
        y_prob = F.softmax(y_pred, dim=1)
        y_true = y_true.float()

        dims = (0, 2, 3)
        intersection = torch.sum(y_prob * y_true, dims)  # (C,)
        cardinality = torch.sum(y_prob + y_true, dims)  # (C,)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss_per_class = 1.0 - dice  # (C,)

        if self.class_weights is not None:
            # ★ ここで y_pred と同じデバイスに載せるのがポイント
            w = torch.tensor(
                self.class_weights, device=y_pred.device, dtype=y_pred.dtype
            )
            loss = (dice_loss_per_class * w).sum() / (w.sum() + self.eps)
        else:
            loss = dice_loss_per_class.mean()

        return loss


class WeightedMultiClassDiceLoss(nn.Module):
    """
    クラス重み付きマルチクラス DiceLoss
    y_pred: ロジット (B, C, H, W)
    y_true: one-hot (B, C, H, W)
    """

    def __init__(self, class_weights=None, eps=1e-7):
        super().__init__()
        self.eps = eps
        if class_weights is None:
            # 背景 < 神経 < 硬膜管 くらいのイメージ（仮）
            class_weights = torch.tensor([0.2, 0.4, 0.4], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

    def forward(self, y_pred, y_true):
        y_prob = F.softmax(y_pred, dim=1)
        y_true = y_true.float()

        dims = (0, 2, 3)
        intersection = torch.sum(y_prob * y_true, dims)  # (C,)
        cardinality = torch.sum(y_prob + y_true, dims)  # (C,)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)  # (C,)

        dice_loss_per_class = 1.0 - dice  # (C,)
        # 重み付き平均
        loss = (dice_loss_per_class * self.class_weights).sum() / (
            self.class_weights.sum() + self.eps
        )
        return loss


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        albu.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        albu.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.HueSaturationValue(p=0.5),
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

        # 0: background, 38: nerve, 75: spinal
        self.class_values = [0, 38, 75]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        masks = np.array([(mask == v) for v in self.class_values])
        mask = np.stack(masks, axis=-1).astype("float32")  # (H, W, 3)

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
# Train / Valid loop
# =========================
def run_one_epoch(model, loader, criterion, metric, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_f = 0.0
    n_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)  # (B, C, H, W)

        batch_size = images.size(0)
        n_samples += batch_size

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)  # (B, C, H, W)
            loss = criterion(logits, masks)
            f_val = metric(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * batch_size
        running_f += f_val.item() * batch_size

    epoch_loss = running_loss / max(1, n_samples)
    epoch_f = running_f / max(1, n_samples)

    return {"dice_loss": epoch_loss, "fscore": epoch_f}


# =========================
# Main
# =========================
def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        os.path.join(TRAIN_DIR, "image"),
        os.path.join(TRAIN_DIR, "mask"),
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    valid_dataset = Dataset(
        os.path.join(VAL_DIR, "image"),
        os.path.join(VAL_DIR, "mask"),
        augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    ).to(DEVICE)

    # 例：背景の重みを軽く、神経・脊髄を重く
    criterion = MultiClassDiceLoss(
        class_weights=[0.2, 1.0, 2.0], eps=EPS  # [background, nerve, spinal]
    )

    metric = FscoreMetric(ignore_background=True, eps=EPS)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

    max_score = 0.0
    x_epoch_data = []
    train_dice_loss = []
    train_f_score = []
    valid_dice_loss = []
    valid_f_score = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch}")

        train_logs = run_one_epoch(
            model, train_loader, criterion, metric, optimizer, DEVICE, train=True
        )
        val_logs = run_one_epoch(
            model, valid_loader, criterion, metric, optimizer, DEVICE, train=False
        )

        print(
            f"  train: loss={train_logs['dice_loss']:.4f}, f={train_logs['fscore']:.4f}"
        )
        print(f"  valid: loss={val_logs['dice_loss']:.4f}, f={val_logs['fscore']:.4f}")

        x_epoch_data.append(epoch)
        train_dice_loss.append(train_logs["dice_loss"])
        train_f_score.append(train_logs["fscore"])
        valid_dice_loss.append(val_logs["dice_loss"])
        valid_f_score.append(val_logs["fscore"])

        if max_score < val_logs["fscore"]:
            max_score = val_logs["fscore"]
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model, SAVE_PATH)
            print(f"Model saved: {SAVE_PATH}")

        if epoch == 20:
            optimizer.param_groups[0]["lr"] = LR_AFTER_20
            print("Decrease learning rate to 1e-4!")

    # 可視化
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_epoch_data, train_dice_loss, label="train")
    ax1.plot(x_epoch_data, valid_dice_loss, label="validation")
    ax1.set_title("dice loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("dice_loss")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x_epoch_data, train_f_score, label="train")
    ax2.plot(x_epoch_data, valid_f_score, label="validation")
    ax2.set_title("fscore")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("fscore")
    ax2.legend(loc="upper left")

    plt.show()


if __name__ == "__main__":
    main()
