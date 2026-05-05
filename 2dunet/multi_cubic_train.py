# train_3d_multi_isotropic.py
import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ★ 等方化に必要
try:
    import scipy.ndimage as ndi
except ImportError as e:
    raise ImportError("等方化(resampling)に scipy が必要です: pip install scipy") from e


# ============================
# 3D U-Net (Encoder + shared decoder)
# ============================


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskUNet3D(nn.Module):
    """
    3Dマルチタスク U-Net
      - Encoder〜低解像度側 Decoder を共有
      - 高解像度側 Decoder を 神経根 / 硬膜管 で分岐
    出力:
      out_root: 神経根 (B,1,D,H,W)
      out_dura: 硬膜管 (B,1,D,H,W)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)

        # shared decoder (coarse)
        self.up4 = nn.ConvTranspose3d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        # root branch (fine)
        self.up1_root = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1_root = DoubleConv3D(base_channels * 2, base_channels)
        self.out_root = nn.Conv3d(base_channels, 1, kernel_size=1)

        # dura branch (fine)
        self.up1_dura = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1_dura = DoubleConv3D(base_channels * 2, base_channels)
        self.out_dura = nn.Conv3d(base_channels, 1, kernel_size=1)

    def _center_crop_to(
        self, enc: torch.Tensor, ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, d_ref, h_ref, w_ref = ref.size()
        _, _, d_enc, h_enc, w_enc = enc.size()

        d_target = min(d_ref, d_enc)
        h_target = min(h_ref, h_enc)
        w_target = min(w_ref, w_enc)

        d_start = (d_enc - d_target) // 2
        h_start = (h_enc - h_target) // 2
        w_start = (w_enc - w_target) // 2

        enc_c = enc[
            :,
            :,
            d_start : d_start + d_target,
            h_start : h_start + h_target,
            w_start : w_start + w_target,
        ]

        if (d_ref, h_ref, w_ref) != (d_target, h_target, w_target):
            d_start_r = (d_ref - d_target) // 2
            h_start_r = (h_ref - h_target) // 2
            w_start_r = (w_ref - w_target) // 2
            ref = ref[
                :,
                :,
                d_start_r : d_start_r + d_target,
                h_start_r : h_start_r + h_target,
                w_start_r : w_start_r + w_target,
            ]

        return enc_c, ref

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # shared decoder
        u4 = self.up4(b)
        e4_c, u4 = self._center_crop_to(e4, u4)
        d4 = self.dec4(torch.cat([u4, e4_c], dim=1))

        u3 = self.up3(d4)
        e3_c, u3 = self._center_crop_to(e3, u3)
        d3 = self.dec3(torch.cat([u3, e3_c], dim=1))

        u2 = self.up2(d3)
        e2_c, u2 = self._center_crop_to(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2_c], dim=1))

        # root branch
        u1r = self.up1_root(d2)
        e1_cr, u1r = self._center_crop_to(e1, u1r)
        d1r = self.dec1_root(torch.cat([u1r, e1_cr], dim=1))
        out_root = self.out_root(d1r)

        # dura branch
        u1d = self.up1_dura(d2)
        e1_cd, u1d = self._center_crop_to(e1, u1d)
        d1d = self.dec1_dura(torch.cat([u1d, e1_cd], dim=1))
        out_dura = self.out_dura(d1d)

        return out_root, out_dura


# ============================
# Resampling (isotropic)
# ============================


def resample_to_spacing(
    img: np.ndarray,
    lab: np.ndarray,
    orig_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    img: (X,Y,Z) float32
    lab: (X,Y,Z) int16
    orig_spacing(mm) -> target_spacing(mm) へ resample
    """
    sx, sy, sz = orig_spacing
    tx, ty, tz = target_spacing

    zoom_factors = (sx / tx, sy / ty, sz / tz)

    # 画像は三次補間、ラベルは最近傍
    img_r = ndi.zoom(img, zoom=zoom_factors, order=3).astype(np.float32)
    lab_r = ndi.zoom(lab, zoom=zoom_factors, order=0).astype(np.int16)
    return img_r, lab_r


# ============================
# Dataset (3D volume / multi-task) + fixed crop + isotropic + optional augment
# ============================


class Nifti3DDataset(Dataset):
    """
    1症例 = 1サンプル（3D volumeそのまま）
      ch0: 神経根
      ch1: 硬膜管

    前処理フロー:
      load -> fixed crop -> resample to target spacing -> normalize -> masks -> (augment) -> tensor
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        nerve_root_label: int = 1,
        dura_label: int = 2,
        augment: bool = False,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        crop_x: Tuple[int, int] = (50, 200),
        crop_y: Tuple[int, int] = (45, 210),
    ):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.nerve_root_label = nerve_root_label
        self.dura_label = dura_label
        self.augment = augment
        self.target_spacing = (
            float(target_spacing[0]),
            float(target_spacing[1]),
            float(target_spacing[2]),
        )
        self.crop_x = crop_x
        self.crop_y = crop_y

        self.case_ids = [
            os.path.splitext(os.path.basename(p))[0] for p in self.label_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def apply_augment(self, img, rmask, dmask):
        # img: (1, X, Y, Z), masks: (X,Y,Z)
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :, :]
            rmask = rmask[::-1, :, :]
            dmask = dmask[::-1, :, :]

        if np.random.rand() < 0.5:
            img = img[:, :, ::-1, :]
            rmask = rmask[:, ::-1, :]
            dmask = dmask[:, ::-1, :]

        if np.random.rand() < 0.5:
            scale = 0.9 + 0.2 * np.random.rand()
            img = np.clip(img * scale, 0.0, 1.0)

        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.03, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        # stride問題回避
        img = img.astype(np.float32, copy=True)
        rmask = rmask.astype(np.float32, copy=True)
        dmask = dmask.astype(np.float32, copy=True)
        return img, rmask, dmask

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        lab_path = self.label_paths[idx]

        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        img = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)
        lab = lab_nii.get_fdata().astype(np.int16)

        if img.shape != lab.shape:
            raise ValueError(
                f"Shape mismatch: img {img.shape} vs lab {lab.shape} for {img_path}"
            )

        # ===== fixed crop（まずは現状のROI指定を維持）=====
        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
        img = img[x0:x1, y0:y1, :]
        lab = lab[x0:x1, y0:y1, :]
        # ===============================================

        # ★ spacing を header から取得（元画像のspacing）
        zooms = img_nii.header.get_zooms()[:3]
        orig_spacing = (
            float(zooms[0]),
            float(zooms[1]),
            float(zooms[2]),
        )  # 例: (1.25,1.25,3.0)

        # ★ 等方化（target_spacing へ）
        img, lab = resample_to_spacing(
            img, lab, orig_spacing=orig_spacing, target_spacing=self.target_spacing
        )

        # normalize 0-1（等方化後）
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        # channel付与: (1,X,Y,Z)
        img = img[None, ...].astype(np.float32, copy=True)

        # masks
        root_mask = (lab == self.nerve_root_label).astype(np.float32)
        dura_mask = (lab == self.dura_label).astype(np.float32)

        if self.augment:
            img, root_mask, dura_mask = self.apply_augment(img, root_mask, dura_mask)
        else:
            img = img.astype(np.float32, copy=True)
            root_mask = root_mask.astype(np.float32, copy=True)
            dura_mask = dura_mask.astype(np.float32, copy=True)

        mask = np.stack([root_mask, dura_mask], axis=0).astype(
            np.float32, copy=True
        )  # (2,X,Y,Z)

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        return img_tensor, mask_tensor, self.case_ids[idx]


# ============================
# Loss & Dice
# ============================


def center_crop_5d_to_match(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == 5 and b.dim() == 5
    _, _, d_a, h_a, w_a = a.shape
    _, _, d_b, h_b, w_b = b.shape

    d_t = min(d_a, d_b)
    h_t = min(h_a, h_b)
    w_t = min(w_a, w_b)

    def crop(t, d_t, h_t, w_t):
        _, _, d, h, w = t.shape
        d_s = (d - d_t) // 2
        h_s = (h - h_t) // 2
        w_s = (w - w_t) // 2
        return t[:, :, d_s : d_s + d_t, h_s : h_s + h_t, w_s : w_s + w_t]

    return crop(a, d_t, h_t, w_t), crop(b, d_t, h_t, w_t)


def dice_loss_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.contiguous().view(probs.size(0), -1)
    targets_flat = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps
    dice = 2.0 * intersection / denom
    return 1.0 - dice.mean()


def combined_loss_single(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_aligned, targets_aligned = center_crop_5d_to_match(logits, targets)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits_aligned, targets_aligned
    )
    dsc = dice_loss_from_logits(logits_aligned, targets_aligned)
    return bce + dsc


def multitask_loss(
    logits_root: torch.Tensor,
    logits_dura: torch.Tensor,
    masks: torch.Tensor,
    lambda_root: float = 1.0,
    lambda_dura: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root_targets = masks[:, 0:1, ...]
    dura_targets = masks[:, 1:2, ...]
    loss_root = combined_loss_single(logits_root, root_targets)
    loss_dura = combined_loss_single(logits_dura, dura_targets)
    loss = lambda_root * loss_root + lambda_dura * loss_dura
    return loss, loss_root, loss_dura


# ============================
# Train / Val loops
# ============================


def train_one_epoch(model, loader, optimizer, device, lambda_root=1.0, lambda_dura=0.3):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits_root, logits_dura = model(imgs)
        loss, _, _ = multitask_loss(
            logits_root,
            logits_dura,
            masks,
            lambda_root=lambda_root,
            lambda_dura=lambda_dura,
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, lambda_root=1.0, lambda_dura=0.3):
    model.eval()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits_root, logits_dura = model(imgs)
        loss, _, _ = multitask_loss(
            logits_root,
            logits_dura,
            masks,
            lambda_root=lambda_root,
            lambda_dura=lambda_dura,
        )
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


# ============================
# Utility: pairing & split
# ============================


def pair_tr_paths(dataset_root: str) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(dataset_root, "imagesTr")
    lab_dir = os.path.join(dataset_root, "labelsTr")

    label_files = [
        f for f in os.listdir(lab_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    label_files.sort()

    image_paths, label_paths = [], []
    for lf in label_files:
        case_id = lf.replace(".nii.gz", "").replace(".nii", "")
        img_name = f"{case_id}_0000.nii.gz"
        img_path = os.path.join(img_dir, img_name)
        lab_path = os.path.join(lab_dir, lf)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found for label {lf}: expected {img_path}"
            )

        image_paths.append(img_path)
        label_paths.append(lab_path)
    return image_paths, label_paths


def train_val_split(
    image_paths: List[str],
    label_paths: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
):
    assert len(image_paths) == len(label_paths)
    n = len(image_paths)
    indices = list(range(n))

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def subset(lst, idxs):
        return [lst[i] for i in idxs]

    return (
        subset(image_paths, train_idx),
        subset(label_paths, train_idx),
        subset(image_paths, val_idx),
        subset(label_paths, val_idx),
    )


# ============================
# main
# ============================


def main():
    parser = argparse.ArgumentParser(
        description="3D Multi-task U-Net (root + dura) + fixed crop + isotropic resampling"
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--nerve_root_label", type=int, default=1)
    parser.add_argument("--dura_label", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="./ckpt3d_mt")
    parser.add_argument("--num_workers", type=int, default=2)

    # loss weight
    parser.add_argument("--lambda_root", type=float, default=1.0)
    parser.add_argument("--lambda_dura", type=float, default=0.3)

    # save
    parser.add_argument("--save_name", type=str, default="best_3dunet")

    # ★ 等方化ターゲット spacing（mm）
    parser.add_argument(
        "--target_spacing_mm",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="resample target spacing (mm), e.g. 1.0 1.0 1.0",
    )

    # fixed crop
    parser.add_argument("--crop_x", type=int, nargs=2, default=[50, 200])
    parser.add_argument("--crop_y", type=int, nargs=2, default=[45, 210])

    # augment
    parser.add_argument(
        "--augment", action="store_true", help="enable augmentation for train only"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Target spacing (mm):", tuple(args.target_spacing_mm))
    print("Fixed crop X:", tuple(args.crop_x), "Y:", tuple(args.crop_y))
    print("Augment:", args.augment)

    # Tr
    tr_imgs, tr_labs = pair_tr_paths(args.dataset_root)
    print(f"#Total Tr cases: {len(tr_imgs)}")

    train_imgs, train_labs, val_imgs, val_labs = train_val_split(
        tr_imgs, tr_labs, val_ratio=args.val_ratio, seed=42
    )
    print(f"#Train cases: {len(train_imgs)}, #Val cases: {len(val_imgs)}")

    train_ds = Nifti3DDataset(
        train_imgs,
        train_labs,
        nerve_root_label=args.nerve_root_label,
        dura_label=args.dura_label,
        augment=args.augment,
        target_spacing=tuple(args.target_spacing_mm),
        crop_x=tuple(args.crop_x),
        crop_y=tuple(args.crop_y),
    )

    val_ds = Nifti3DDataset(
        val_imgs,
        val_labs,
        nerve_root_label=args.nerve_root_label,
        dura_label=args.dura_label,
        augment=False,
        target_spacing=tuple(args.target_spacing_mm),
        crop_x=tuple(args.crop_x),
        crop_y=tuple(args.crop_y),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MultiTaskUNet3D(in_channels=1, base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, f"{args.save_name}.pth")

    patience = 30
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_root=args.lambda_root,
            lambda_dura=args.lambda_dura,
        )
        val_loss = validate(
            model,
            val_loader,
            device,
            lambda_root=args.lambda_root,
            lambda_dura=args.lambda_dura,
        )
        print(f"  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "target_spacing_mm": tuple(args.target_spacing_mm),
                    "crop_x": tuple(args.crop_x),
                    "crop_y": tuple(args.crop_y),
                },
                best_path,
            )
            print(f"  >>> Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping: no improvement in val_loss for {patience} consecutive epochs."
                )
                break


if __name__ == "__main__":
    main()
