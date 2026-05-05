# -*- coding: utf-8 -*-
"""
Training script for multi-class segmentation (background / nerve / spinal)
using 2D Axial slices from NIfTI volumes (imagesTr / labelsTr)
with Attention U-Net (2D).
"""

import os
import glob
import traceback
from datetime import datetime

import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

import albumentations as albu
from segmentation_models_pytorch.utils.metrics import Fscore


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = r"C:\Users\orilab\Desktop\masumoto\smp\Dataset001_lumber"
IMAGES_TR_DIR = os.path.join(DATA_ROOT, "imagesTr")
LABELS_TR_DIR = os.path.join(DATA_ROOT, "labelsTr")

SAVE_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\checkpoints_attention"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 200
BATCH_SIZE = 8
LR_INIT = 1e-3
LR_AFTER_20 = 1e-4
IMAGE_SIZE = 256  # もともと 256x256 なのでそのまま
VAL_RATIO = 0.2  # 症例のうち 20% を validation にする

CLASSES = ["background", "nerve", "spinal"]  # 0,1,2


# =========================
# Utility
# =========================
def strip_nii_ext(fname: str) -> str:
    """'case001_0000.nii.gz' -> 'case001_0000'"""
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return os.path.splitext(fname)[0]


def find_train_cases(images_dir, labels_dir):
    """
    imagesTr: caseXXX_0000.nii.gz
    labelsTr: caseXXX.nii.gz または caseXXX.nii
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    cases = {}

    for img_path in image_files:
        base = os.path.basename(img_path)
        stem = strip_nii_ext(base)

        # *_0000 のみ使用
        if not stem.endswith("_0000"):
            continue

        case_id = stem[:-5]  # 'case001_0000' -> 'case001'

        lab_nii_gz = os.path.join(labels_dir, case_id + ".nii.gz")
        lab_nii = os.path.join(labels_dir, case_id + ".nii")

        if os.path.exists(lab_nii_gz):
            lab_path = lab_nii_gz
        elif os.path.exists(lab_nii):
            lab_path = lab_nii
        else:
            print(f"[WARN] label not found for {base}, skip.")
            continue

        img_nii = nib.load(img_path)
        lab_nii_obj = nib.load(lab_path)

        img_vol = img_nii.get_fdata()
        lab_vol = lab_nii_obj.get_fdata()

        # 4D の場合は 1ch目だけ使う
        if img_vol.ndim == 4:
            img_vol = img_vol[..., 0]
        if lab_vol.ndim == 4:
            lab_vol = lab_vol[..., 0]

        img_vol = img_vol.astype(np.float32)
        lab_vol = lab_vol.astype(np.int16)

        H, W, Z = img_vol.shape
        print(
            f"[{case_id}] use image: {base}, label: {os.path.basename(lab_path)}, shape {H}x{W}x{Z}"
        )

        cases[case_id] = {
            "img": img_vol,
            "lab": lab_vol,
        }

    return cases


def get_training_augmentation():
    """
    Albumentations v1.4.x で使える変換のみ使用
    """
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(
                scale_limit=0.5,
                rotate_limit=0,
                shift_limit=0.1,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            albu.PadIfNeeded(
                min_height=IMAGE_SIZE,
                min_width=IMAGE_SIZE,
                always_apply=True,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            albu.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, always_apply=True),
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
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
    )


def get_preprocessing():
    """
    0-1 正規化 + CHW 変換
    """

    def normalize(img, **kwargs):
        # すでに Dataset 側で 0-1 にしていても安全にそのまま
        return img.astype("float32")

    return albu.Compose(
        [
            albu.Lambda(image=normalize),
            albu.Lambda(
                image=lambda x, **k: x.transpose(2, 0, 1).astype("float32"),
                mask=lambda x, **k: x.transpose(2, 0, 1).astype("float32"),
            ),
        ]
    )


# =========================
# Dataset
# =========================
class NiftiSliceDataset(TorchDataset):
    """
    caseごとの 3D volume から (case_id, z) で 2D スライスを取り出す Dataset
    - 画像：各スライスを min-max 正規化 → 3ch グレースケール
    - ラベル：0/1/2 を one-hot [H,W,3]
    """

    def __init__(self, cases_dict, case_ids, augmentation=None, preprocessing=None):
        self.cases_dict = cases_dict
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.slices = []  # list of (case_id, z)
        for cid in case_ids:
            vol = cases_dict[cid]["img"]
            Z = vol.shape[2]
            for z in range(Z):
                self.slices.append((cid, z))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        cid, z = self.slices[idx]
        img_vol = self.cases_dict[cid]["img"]
        lab_vol = self.cases_dict[cid]["lab"]

        img_slice = img_vol[:, :, z].astype(np.float32)
        lab_slice = lab_vol[:, :, z].astype(np.int16)

        # min-max 正規化
        vmin, vmax = img_slice.min(), img_slice.max()
        if vmax > vmin:
            img_norm = (img_slice - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img_slice, dtype=np.float32)

        img_u8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)
        image = np.stack([img_u8, img_u8, img_u8], axis=-1)  # (H,W,3)

        # one-hot mask: 0,1,2
        mask_bg = (lab_slice == 0).astype(np.float32)
        mask_nerve = (lab_slice == 1).astype(np.float32)
        mask_spinal = (lab_slice == 2).astype(np.float32)
        mask = np.stack([mask_bg, mask_nerve, mask_spinal], axis=-1)  # (H,W,3)

        # augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # preprocessing (0-1化 + CHW)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask


# =========================
# Attention U-Net 2D
# =========================
class AttentionBlock2D(nn.Module):
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
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(512, 1024)

        self.att4 = AttentionBlock2D(512, 512, 256)
        self.att3 = AttentionBlock2D(256, 256, 128)
        self.att2 = AttentionBlock2D(128, 128, 64)
        self.att1 = AttentionBlock2D(64, 64, 32)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        center = self.center(p4)

        d4 = self.up4(center)
        e4_att = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        logits = self.seg_head(d1)
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
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", DEVICE)

    # --- case 読み込み ---
    cases_dict = find_train_cases(IMAGES_TR_DIR, LABELS_TR_DIR)
    all_cases = sorted(cases_dict.keys())
    n_cases = len(all_cases)

    if n_cases == 0:
        print("[ERROR] no training cases found.")
        return

    n_val = max(1, int(n_cases * VAL_RATIO))
    n_train = n_cases - n_val

    train_cases = all_cases[:n_train]
    val_cases = all_cases[n_train:]

    print(f"#cases total: {n_cases}, train: {n_train}, val: {n_val}")
    print("Train cases:", train_cases)
    print("Val   cases:", val_cases)

    # Dataset / DataLoader
    train_dataset = NiftiSliceDataset(
        cases_dict,
        train_cases,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(),
    )
    val_dataset = NiftiSliceDataset(
        cases_dict,
        val_cases,
        augmentation=None,
        preprocessing=get_preprocessing(),
    )

    print(f"#slices train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Model
    model = AttentionUNet2D(in_channels=3, num_classes=len(CLASSES)).to(DEVICE)

    # Loss / optimizer / metric
    weights = torch.tensor([0.1, 1.5, 0.5], device=DEVICE)
    loss = MultiClassDiceLoss(class_weights=weights)

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=LR_INIT),
        ]
    )

    metrics = [Fscore(threshold=0.5)]

    # smp の Trainer を使わず、自前ループにしてもよいけど、
    # そのまま logits を渡しても Fscore 側で softmax/sigmoid を処理してくれる前提。
    from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    val_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Logging
    today_str = datetime.now().strftime("%Y%m%d_%H%M")
    max_score = 0.0

    x_epoch = []
    tr_loss_list, tr_f_list = [], []
    va_loss_list, va_f_list = [], []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch}")
        try:
            train_logs = train_epoch.run(train_loader)
            val_logs = val_epoch.run(val_loader)
        except Exception:
            print("例外が発生しました:")
            traceback.print_exc()
            continue

        x_epoch.append(epoch)
        tr_loss_list.append(train_logs["MultiClassDiceLoss"])
        tr_f_list.append(train_logs["fscore"])
        va_loss_list.append(val_logs["MultiClassDiceLoss"])
        va_f_list.append(val_logs["fscore"])

        # save best
        if val_logs["fscore"] > max_score:
            max_score = val_logs["fscore"]
            filename = f"{today_str}_att_unet2d_nifti2d.pth"
            save_path = os.path.join(SAVE_DIR, filename)
            # state_dict 保存（→ test.py 側で AttentionUNet2D を作って load する）
            torch.save(model.state_dict(), save_path)
            print(f"Model saved: {save_path}")

        # LR 調整
        if epoch == 20:
            optimizer.param_groups[0]["lr"] = LR_AFTER_20
            print("Decrease learning rate to 1e-4!")

    # loss / fscore の曲線
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_epoch, tr_loss_list, label="train")
    ax1.plot(x_epoch, va_loss_list, label="val")
    ax1.set_title("Dice loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x_epoch, tr_f_list, label="train")
    ax2.plot(x_epoch, va_f_list, label="val")
    ax2.set_title("F-score")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("fscore")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    main()
