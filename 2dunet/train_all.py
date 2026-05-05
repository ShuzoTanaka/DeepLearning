# train_3d_multi_isotropic.py
import os
import argparse
from typing import List, Tuple
import csv

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# resample + distance metrics
try:
    import scipy.ndimage as ndi
    from scipy.spatial.distance import cdist
except ImportError as e:
    raise ImportError("scipy が必要です: pip install scipy") from e


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
    sx, sy, sz = orig_spacing
    tx, ty, tz = target_spacing
    zoom_factors = (sx / tx, sy / ty, sz / tz)
    img_r = ndi.zoom(img, zoom=zoom_factors, order=3).astype(np.float32)
    lab_r = ndi.zoom(lab, zoom=zoom_factors, order=0).astype(np.int16)
    return img_r, lab_r


# ============================
# Patch sampling (foreground oversampling)
# ============================


def extract_patch_zyx(
    img_zyx: np.ndarray,
    lab_zyx: np.ndarray,
    patch_size_zyx: Tuple[int, int, int],
    fg_ratio: float,
    fg_labels: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    img_zyx, lab_zyx: (Z,Y,X)
    patch_size_zyx: (pZ,pY,pX)
    """
    Z, Y, X = img_zyx.shape
    pZ, pY, pX = patch_size_zyx

    # fallback if image is smaller than patch
    pZ = min(pZ, Z)
    pY = min(pY, Y)
    pX = min(pX, X)

    if np.random.rand() < fg_ratio:
        fg = np.argwhere(np.isin(lab_zyx, fg_labels))
        if len(fg) > 0:
            cz, cy, cx = fg[np.random.randint(len(fg))]
        else:
            cz, cy, cx = (
                np.random.randint(Z),
                np.random.randint(Y),
                np.random.randint(X),
            )
    else:
        cz, cy, cx = np.random.randint(Z), np.random.randint(Y), np.random.randint(X)

    z0 = int(np.clip(cz - pZ // 2, 0, Z - pZ))
    y0 = int(np.clip(cy - pY // 2, 0, Y - pY))
    x0 = int(np.clip(cx - pX // 2, 0, X - pX))

    img_p = img_zyx[z0 : z0 + pZ, y0 : y0 + pY, x0 : x0 + pX]
    lab_p = lab_zyx[z0 : z0 + pZ, y0 : y0 + pY, x0 : x0 + pX]
    return img_p, lab_p


# ============================
# Dataset (3D volume / multi-task) + fixed crop + (optional) isotropic + (optional) patch+FG + norm switch + augment switch
# ============================


class Nifti3DDataset(Dataset):
    """
    1症例 = 1サンプル（ただし enable_patch=True の場合は 1症例からランダムパッチを返す）
    ch0: 神経根, ch1: 硬膜管
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        nerve_root_label: int = 1,
        dura_label: int = 2,
        enable_augment: bool = False,
        enable_isotropic: bool = True,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        crop_x: Tuple[int, int] = (50, 200),
        crop_y: Tuple[int, int] = (45, 210),
        enable_patch: bool = False,
        patch_size_zyx: Tuple[int, int, int] = (48, 192, 224),
        fg_ratio: float = 0.33,
        enable_zscore_norm: bool = False,
    ):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.nerve_root_label = nerve_root_label
        self.dura_label = dura_label

        self.enable_augment = enable_augment
        self.enable_isotropic = enable_isotropic
        self.target_spacing = tuple(map(float, target_spacing))
        self.crop_x = crop_x
        self.crop_y = crop_y

        self.enable_patch = enable_patch
        self.patch_size_zyx = tuple(map(int, patch_size_zyx))
        self.fg_ratio = float(fg_ratio)

        self.enable_zscore_norm = enable_zscore_norm

        self.case_ids = [
            os.path.basename(p).replace(".nii.gz", "").replace(".nii", "")
            for p in self.label_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def apply_augment(self, img, rmask, dmask):
        # img: (1, Z, Y, X), masks: (Z,Y,X)
        if np.random.rand() < 0.5:
            img = img[:, :, :, ::-1]
            rmask = rmask[:, :, ::-1]
            dmask = dmask[:, :, ::-1]

        if np.random.rand() < 0.5:
            img = img[:, :, ::-1, :]
            rmask = rmask[:, ::-1, :]
            dmask = dmask[:, ::-1, :]

        if np.random.rand() < 0.5:
            scale = 0.9 + 0.2 * np.random.rand()
            img = img * scale

        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.03, size=img.shape).astype(np.float32)
            img = img + noise

        img = img.astype(np.float32, copy=True)
        rmask = rmask.astype(np.float32, copy=True)
        dmask = dmask.astype(np.float32, copy=True)
        return img, rmask, dmask

    def normalize(self, img_zyx: np.ndarray) -> np.ndarray:
        img_zyx = img_zyx.astype(np.float32)
        if self.enable_zscore_norm:
            m = float(img_zyx.mean())
            s = float(img_zyx.std())
            return (img_zyx - m) / (s + 1e-6)
        else:
            vmin = float(img_zyx.min())
            vmax = float(img_zyx.max())
            return (img_zyx - vmin) / (vmax - vmin + 1e-6)

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

        # fixed crop in (X,Y,Z)
        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
        img = img[x0:x1, y0:y1, :]
        lab = lab[x0:x1, y0:y1, :]

        # spacing from original header (X,Y,Z)
        zooms = img_nii.header.get_zooms()[:3]
        orig_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        # isotropic resample (still in X,Y,Z arrays)
        if self.enable_isotropic:
            img, lab = resample_to_spacing(
                img, lab, orig_spacing=orig_spacing, target_spacing=self.target_spacing
            )

        # convert to (Z,Y,X) for torch Conv3d with (D,H,W)
        img_zyx = np.transpose(img, (2, 1, 0))  # (Z,Y,X)
        lab_zyx = np.transpose(lab, (2, 1, 0))  # (Z,Y,X)

        # patch + FG oversampling (train/val option)
        if self.enable_patch:
            img_zyx, lab_zyx = extract_patch_zyx(
                img_zyx,
                lab_zyx,
                patch_size_zyx=self.patch_size_zyx,
                fg_ratio=self.fg_ratio,
                fg_labels=[self.nerve_root_label, self.dura_label],
            )

        # normalize
        img_zyx = self.normalize(img_zyx)

        # channel add: (1,Z,Y,X)
        img_c = img_zyx[None, ...].astype(np.float32, copy=True)

        # masks
        root_mask = (lab_zyx == self.nerve_root_label).astype(np.float32)
        dura_mask = (lab_zyx == self.dura_label).astype(np.float32)

        if self.enable_augment:
            img_c, root_mask, dura_mask = self.apply_augment(
                img_c, root_mask, dura_mask
            )
        else:
            img_c = img_c.astype(np.float32, copy=True)
            root_mask = root_mask.astype(np.float32, copy=True)
            dura_mask = dura_mask.astype(np.float32, copy=True)

        mask = np.stack([root_mask, dura_mask], axis=0).astype(
            np.float32, copy=True
        )  # (2,Z,Y,X)

        img_tensor = torch.from_numpy(img_c)  # (1,Z,Y,X)
        mask_tensor = torch.from_numpy(mask)  # (2,Z,Y,X)

        return img_tensor, mask_tensor, self.case_ids[idx]


# ============================
# Loss
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


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    running_loss = 0.0

    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits_root, logits_dura = model(imgs)

        loss, _, _ = compute_loss(
            logits_root,
            logits_dura,
            masks,
            loss_mode=args.loss_mode,
            lambda_root=args.lambda_root,
            lambda_dura=args.lambda_dura,
        )

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits_root, logits_dura = model(imgs)
            loss, _, _ = compute_loss(
                logits_root,
                logits_dura,
                masks,
                loss_mode=args.loss_mode,
                lambda_root=args.lambda_root,
                lambda_dura=args.lambda_dura,
            )
            running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


# ============================
# Metrics (Dice / HD95 / ASD) per case with NIfTI spacing
# ============================


def center_crop_3d_pair_to_min(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    a, b: (Z,Y,X)
    両者を中心cropして、共通の最小形状に揃える
    """
    za, ya, xa = a.shape
    zb, yb, xb = b.shape

    zt = min(za, zb)
    yt = min(ya, yb)
    xt = min(xa, xb)

    def crop(x, zt, yt, xt):
        z, y, xw = x.shape
        zs = (z - zt) // 2
        ys = (y - yt) // 2
        xs = (xw - xt) // 2
        return x[zs : zs + zt, ys : ys + yt, xs : xs + xt]

    return crop(a, zt, yt, xt), crop(b, zt, yt, xt)


def dice_score_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2.0 * inter / (denom + eps))


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """
    get surface voxel coordinates (N,3) in zyx index space
    simple morphological gradient by erosion
    """
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.int32)
    eroded = ndi.binary_erosion(mask, iterations=1)
    surf = np.logical_and(mask, np.logical_not(eroded))
    coords = np.argwhere(surf)  # (N,3) in zyx
    return coords.astype(np.int32)


def hd95_asd_mm(
    pred: np.ndarray, gt: np.ndarray, spacing_zyx: Tuple[float, float, float]
) -> Tuple[float, float]:
    """
    surface distance based HD95 / ASD in mm.
    spacing_zyx corresponds to (z,y,x) voxel spacing.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan"), float("nan")

    p = surface_voxels(pred)
    g = surface_voxels(gt)
    if len(p) == 0 or len(g) == 0:
        return float("nan"), float("nan")

    sp = np.array(spacing_zyx, dtype=np.float32)[None, :]  # (1,3)
    p_mm = p.astype(np.float32) * sp
    g_mm = g.astype(np.float32) * sp

    # pairwise distances
    d = cdist(p_mm, g_mm)  # (Np,Ng)
    d_p = d.min(axis=1)  # from pred->gt
    d_g = d.min(axis=0)  # from gt->pred
    all_d = np.concatenate([d_p, d_g], axis=0)

    hd95 = float(np.percentile(all_d, 95))
    asd = float(all_d.mean())
    return hd95, asd


def compute_loss(
    logits_root: torch.Tensor,
    logits_dura: torch.Tensor,
    masks: torch.Tensor,
    loss_mode: str,
    lambda_root: float = 1.0,
    lambda_dura: float = 0.3,
):
    root_targets = masks[:, 0:1, ...]

    if loss_mode == "root_only":
        loss = combined_loss_single(logits_root, root_targets)
        return loss, loss, torch.tensor(0.0, device=loss.device)

    elif loss_mode == "multitask":
        dura_targets = masks[:, 1:2, ...]
        loss_root = combined_loss_single(logits_root, root_targets)
        loss_dura = combined_loss_single(logits_dura, dura_targets)
        loss = lambda_root * loss_root + lambda_dura * loss_dura
        return loss, loss_root, loss_dura

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")


@torch.no_grad()
def run_test_and_save_csv(
    model: nn.Module,
    imagesTs_dir: str,
    labelsTs_dir: str,
    out_csv_path: str,
    device: torch.device,
    crop_x: Tuple[int, int],
    crop_y: Tuple[int, int],
    enable_isotropic: bool,
    target_spacing: Tuple[float, float, float],
    enable_zscore_norm: bool,
):
    """
    Test: whole volume inference per case
    - no patch (full volume)
    - no augment
    - compute metrics using ORIGINAL NIfTI spacing (after transpose to zyx)
    - save CSV: case_id, Dice, HD95_mm, ASD_mm + MEAN row
    """
    model.eval()

    # list cases from labelsTs
    label_files = [
        f
        for f in os.listdir(labelsTs_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    label_files.sort()

    rows = []
    dices, hd95s, asds = [], [], []

    # ============================
    # Create prediction output dir
    # ============================
    pred_dir = os.path.join(os.path.dirname(out_csv_path), "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for lf in tqdm(label_files, desc="Test"):
        case_id = lf.replace(".nii.gz", "").replace(".nii", "")
        img_path = os.path.join(imagesTs_dir, f"{case_id}_0000.nii.gz")
        lab_path = os.path.join(labelsTs_dir, lf)

        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        img = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)
        lab = lab_nii.get_fdata().astype(np.int16)

        # ----------------------------
        # crop (X,Y)
        # ----------------------------
        x0, x1 = crop_x
        y0, y1 = crop_y
        img = img[x0:x1, y0:y1, :]
        lab = lab[x0:x1, y0:y1, :]

        # spacing
        zooms = img_nii.header.get_zooms()[:3]
        orig_spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        # isotropic
        if enable_isotropic:
            img, lab = resample_to_spacing(img, lab, orig_spacing_xyz, target_spacing)
            spacing_xyz_for_metrics = target_spacing
        else:
            spacing_xyz_for_metrics = orig_spacing_xyz

        # to (Z,Y,X)
        img_zyx = np.transpose(img, (2, 1, 0))
        lab_zyx = np.transpose(lab, (2, 1, 0))

        # normalize
        if enable_zscore_norm:
            m, s = img_zyx.mean(), img_zyx.std()
            img_zyx = (img_zyx - m) / (s + 1e-6)
        else:
            vmin, vmax = img_zyx.min(), img_zyx.max()
            img_zyx = (img_zyx - vmin) / (vmax - vmin + 1e-6)

        x = torch.from_numpy(img_zyx[None, None, ...].astype(np.float32)).to(device)

        # ============================
        # Inference (root + dura)
        # ============================
        logits_root, logits_dura = model(x)

        prob_root = torch.sigmoid(logits_root)[0, 0].cpu().numpy()
        prob_dura = torch.sigmoid(logits_dura)[0, 0].cpu().numpy()

        pred_root = prob_root > 0.5
        pred_dura = prob_dura > 0.5

        # ============================
        # Save predictions as NIfTI
        # ============================
        affine = img_nii.affine

        pred_root_xyz = np.transpose(pred_root.astype(np.uint8), (2, 1, 0))
        pred_dura_xyz = np.transpose(pred_dura.astype(np.uint8), (2, 1, 0))

        nib.save(
            nib.Nifti1Image(pred_root_xyz, affine),
            os.path.join(pred_dir, f"{case_id}_pred_root.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(pred_dura_xyz, affine),
            os.path.join(pred_dir, f"{case_id}_pred_dura.nii.gz"),
        )

        # ============================
        # Metrics (root only)
        # ============================
        gt_root = lab_zyx == 1

        # ★ pred と gt を同じ形状に揃える（中心crop）
        pred_root_c, gt_root_c = center_crop_3d_pair_to_min(pred_root, gt_root)

        sx, sy, sz = spacing_xyz_for_metrics
        spacing_zyx = (float(sz), float(sy), float(sx))

        dsc = dice_score_binary(pred_root_c, gt_root_c)
        hd, asd = hd95_asd_mm(pred_root_c, gt_root_c, spacing_zyx)

        rows.append([case_id, dsc, hd, asd])
        dices.append(dsc)
        hd95s.append(hd)
        asds.append(asd)

    # MEAN row (nan-aware)
    mean_d = float(np.nanmean(np.array(dices, dtype=np.float32)))
    mean_h = float(np.nanmean(np.array(hd95s, dtype=np.float32)))
    mean_a = float(np.nanmean(np.array(asds, dtype=np.float32)))
    rows.append(["MEAN", mean_d, mean_h, mean_a])

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "Dice", "HD95_mm", "ASD_mm"])
        for r in rows:
            w.writerow(r)


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
    n_val = max(1, int(n * val_ratio))
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
        description="3D Multi-task U-Net (root + dura) with optional isotropic / patch+FG oversampling / z-score / augment + test csv"
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

    # switches
    parser.add_argument(
        "--enable_isotropic", action="store_true", help="enable isotropic resampling"
    )
    parser.add_argument(
        "--enable_patch",
        action="store_true",
        help="enable patch training + foreground oversampling",
    )
    parser.add_argument(
        "--enable_zscore_norm",
        action="store_true",
        help="use z-score normalization (else min-max)",
    )
    parser.add_argument(
        "--enable_augment",
        action="store_true",
        help="enable augmentation for train only",
    )

    # isotropic target spacing
    parser.add_argument(
        "--target_spacing_mm", type=float, nargs=3, default=[1.0, 1.0, 1.0]
    )

    # patch params
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[48, 192, 224],
        help="patch size in Z Y X",
    )
    parser.add_argument(
        "--fg_ratio",
        type=float,
        default=0.33,
        help="foreground sampling ratio for patches",
    )

    # fixed crop (X,Y)
    parser.add_argument("--crop_x", type=int, nargs=2, default=[50, 200])
    parser.add_argument("--crop_y", type=int, nargs=2, default=[45, 210])

    # test directories (nnU-Net風Dataset layout想定)
    parser.add_argument(
        "--imagesTs_dir",
        type=str,
        default=None,
        help="optional: override test images dir",
    )
    parser.add_argument(
        "--labelsTs_dir",
        type=str,
        default=None,
        help="optional: override test labels dir",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="multitask",
        choices=["root_only", "multitask"],
        help="loss mode: root_only or multitask (root + dura)",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print(
        "enable_isotropic:",
        args.enable_isotropic,
        "target_spacing_mm:",
        tuple(args.target_spacing_mm),
    )
    print(
        "enable_patch:",
        args.enable_patch,
        "patch_size(Z,Y,X):",
        tuple(args.patch_size),
        "fg_ratio:",
        args.fg_ratio,
    )
    print("enable_zscore_norm:", args.enable_zscore_norm)
    print("enable_augment:", args.enable_augment)
    print("crop_x:", tuple(args.crop_x), "crop_y:", tuple(args.crop_y))

    # Train/Val
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
        enable_augment=args.enable_augment,
        enable_isotropic=args.enable_isotropic,
        target_spacing=tuple(args.target_spacing_mm),
        crop_x=tuple(args.crop_x),
        crop_y=tuple(args.crop_y),
        enable_patch=args.enable_patch,
        patch_size_zyx=tuple(args.patch_size),
        fg_ratio=args.fg_ratio,
        enable_zscore_norm=args.enable_zscore_norm,
    )

    val_ds = Nifti3DDataset(
        val_imgs,
        val_labs,
        nerve_root_label=args.nerve_root_label,
        dura_label=args.dura_label,
        enable_augment=False,
        enable_isotropic=args.enable_isotropic,
        target_spacing=tuple(args.target_spacing_mm),
        crop_x=tuple(args.crop_x),
        crop_y=tuple(args.crop_y),
        enable_patch=args.enable_patch,  # valも同じ設定にする（比較を揃える）
        patch_size_zyx=tuple(args.patch_size),
        fg_ratio=args.fg_ratio,  # valでは fg_ratio は意味薄いが一応同じ
        enable_zscore_norm=args.enable_zscore_norm,
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
            args,
        )

        val_loss = validate(
            model,
            val_loader,
            device,
            args,
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
                    "args": vars(args),
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

    # ============================
    # TEST (after training)
    # ============================
    print("=== TEST (after training) ===")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    imagesTs_dir = args.imagesTs_dir or os.path.join(args.dataset_root, "imagesTs")
    labelsTs_dir = args.labelsTs_dir or os.path.join(args.dataset_root, "labelsTs")
    out_csv = os.path.join(args.out_dir, "test_metrics.csv")

    if not os.path.isdir(imagesTs_dir) or not os.path.isdir(labelsTs_dir):
        print("WARNING: imagesTs/labelsTs not found. Skipping test.")
        print("  imagesTs_dir:", imagesTs_dir)
        print("  labelsTs_dir:", labelsTs_dir)
        return

    run_test_and_save_csv(
        model=model,
        imagesTs_dir=imagesTs_dir,
        labelsTs_dir=labelsTs_dir,
        out_csv_path=out_csv,
        device=device,
        crop_x=tuple(args.crop_x),
        crop_y=tuple(args.crop_y),
        enable_isotropic=args.enable_isotropic,
        target_spacing=tuple(args.target_spacing_mm),
        enable_zscore_norm=args.enable_zscore_norm,
    )
    print(f"Saved test metrics to: {out_csv}")


if __name__ == "__main__":
    main()
