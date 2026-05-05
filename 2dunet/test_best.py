# test.py
import os
import argparse
import csv
from typing import Tuple, List

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn

# scipy
try:
    import scipy.ndimage as ndi
    from scipy.spatial.distance import cdist
except ImportError as e:
    raise ImportError("scipy が必要です: pip install scipy") from e


# ============================
# Model definitions (must match training)
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

        # shared decoder
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

        # root branch
        self.up1_root = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1_root = DoubleConv3D(base_channels * 2, base_channels)
        self.out_root = nn.Conv3d(base_channels, 1, kernel_size=1)

        # dura branch
        self.up1_dura = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1_dura = DoubleConv3D(base_channels * 2, base_channels)
        self.out_dura = nn.Conv3d(base_channels, 1, kernel_size=1)

    def _center_crop_to(self, enc: torch.Tensor, ref: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)

        u4 = self.up4(b)
        e4_c, u4 = self._center_crop_to(e4, u4)
        d4 = self.dec4(torch.cat([u4, e4_c], dim=1))

        u3 = self.up3(d4)
        e3_c, u3 = self._center_crop_to(e3, u3)
        d3 = self.dec3(torch.cat([u3, e3_c], dim=1))

        u2 = self.up2(d3)
        e2_c, u2 = self._center_crop_to(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2_c], dim=1))

        u1r = self.up1_root(d2)
        e1_cr, u1r = self._center_crop_to(e1, u1r)
        d1r = self.dec1_root(torch.cat([u1r, e1_cr], dim=1))
        out_root = self.out_root(d1r)

        u1d = self.up1_dura(d2)
        e1_cd, u1d = self._center_crop_to(e1, u1d)
        d1d = self.dec1_dura(torch.cat([u1d, e1_cd], dim=1))
        out_dura = self.out_dura(d1d)

        return out_root, out_dura


# ============================
# Metrics helpers
# ============================


def center_crop_3d_pair_to_min(a: np.ndarray, b: np.ndarray):
    za, ya, xa = a.shape
    zb, yb, xb = b.shape
    zt, yt, xt = min(za, zb), min(ya, yb), min(xa, xb)

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
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.int32)
    eroded = ndi.binary_erosion(mask, iterations=1)
    surf = np.logical_and(mask, np.logical_not(eroded))
    coords = np.argwhere(surf)  # (N,3) in zyx
    return coords.astype(np.int32)


def hd95_asd_mm(
    pred: np.ndarray, gt: np.ndarray, spacing_zyx: Tuple[float, float, float]
):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan"), float("nan")

    p = surface_voxels(pred)
    g = surface_voxels(gt)
    if len(p) == 0 or len(g) == 0:
        return float("nan"), float("nan")

    sp = np.array(spacing_zyx, dtype=np.float32)[None, :]
    p_mm = p.astype(np.float32) * sp
    g_mm = g.astype(np.float32) * sp

    d = cdist(p_mm, g_mm)
    d_p = d.min(axis=1)
    d_g = d.min(axis=0)
    all_d = np.concatenate([d_p, d_g], axis=0)

    hd95 = float(np.percentile(all_d, 95))
    asd = float(all_d.mean())
    return hd95, asd


def zscore_norm(img_zyx: np.ndarray) -> np.ndarray:
    img_zyx = img_zyx.astype(np.float32)
    m = float(img_zyx.mean())
    s = float(img_zyx.std())
    return (img_zyx - m) / (s + 1e-6)


# ============================
# Main test loop
# ============================


@torch.no_grad()
def run_test(
    model: nn.Module,
    imagesTs_dir: str,
    labelsTs_dir: str,
    out_dir: str,
    device: torch.device,
    crop_x: Tuple[int, int],
    crop_y: Tuple[int, int],
    nerve_root_label: int = 1,
    threshold: float = 0.5,
):
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    label_files = [
        f
        for f in os.listdir(labelsTs_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    label_files.sort()

    rows = []
    dices, hd95s, asds = [], [], []

    for lf in tqdm(label_files, desc="Test"):
        case_id = lf.replace(".nii.gz", "").replace(".nii", "")
        img_path = os.path.join(imagesTs_dir, f"{case_id}_0000.nii.gz")
        lab_path = os.path.join(labelsTs_dir, lf)

        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        img = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)
        lab = lab_nii.get_fdata().astype(np.int16)

        # crop in (X,Y,Z)
        x0, x1 = crop_x
        y0, y1 = crop_y
        img = img[x0:x1, y0:y1, :]
        lab = lab[x0:x1, y0:y1, :]

        # spacing (XYZ) -> for metrics we use original spacing since "no isotropic"
        zooms = img_nii.header.get_zooms()[:3]
        sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
        spacing_zyx = (sz, sy, sx)

        # to (Z,Y,X)
        img_zyx = np.transpose(img, (2, 1, 0))
        lab_zyx = np.transpose(lab, (2, 1, 0))

        # z-score normalization (as requested)
        img_zyx = zscore_norm(img_zyx)

        # model input (1,1,Z,Y,X)
        x = torch.from_numpy(img_zyx[None, None, ...].astype(np.float32)).to(device)

        logits_root, logits_dura = model(x)
        prob_root = torch.sigmoid(logits_root)[0, 0].cpu().numpy()
        prob_dura = torch.sigmoid(logits_dura)[0, 0].cpu().numpy()

        pred_root = prob_root > threshold
        pred_dura = prob_dura > threshold

        # save predictions as nifti (XYZ orientation)
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

        # metrics: ROOT only (label=1)
        gt_root = lab_zyx == nerve_root_label

        # align shapes (center crop)
        pred_root_c, gt_root_c = center_crop_3d_pair_to_min(pred_root, gt_root)

        dsc = dice_score_binary(pred_root_c, gt_root_c)
        hd, asd = hd95_asd_mm(pred_root_c, gt_root_c, spacing_zyx)

        rows.append([case_id, dsc, hd, asd])
        dices.append(dsc)
        hd95s.append(hd)
        asds.append(asd)

    mean_d = float(np.nanmean(np.array(dices, dtype=np.float32)))
    mean_h = float(np.nanmean(np.array(hd95s, dtype=np.float32)))
    mean_a = float(np.nanmean(np.array(asds, dtype=np.float32)))
    rows.append(["MEAN", mean_d, mean_h, mean_a])

    out_csv = os.path.join(out_dir, "test_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "Dice", "HD95_mm", "ASD_mm"])
        w.writerows(rows)

    print("Saved:", out_csv)
    print("Predictions:", pred_dir)


def load_model_from_ckpt(ckpt_path: str, device: torch.device, base_channels: int = 16):
    model = MultiTaskUNet3D(in_channels=1, base_channels=base_channels).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # train_all.py 形式: {"model_state_dict": ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # state_dict直保存の場合にも対応
        model.load_state_dict(ckpt)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(
        description="Test script for MultiTaskUNet3D (no isotropic, z-score, full volume inference)"
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="nnU-Net style dataset root containing imagesTs/labelsTs",
    )
    p.add_argument("--ckpt", type=str, required=True, help="path to saved .pth")
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="output dir to save csv and predictions",
    )
    p.add_argument("--base_channels", type=int, default=16)

    # crop must match training
    p.add_argument("--crop_x", type=int, nargs=2, default=[50, 200])
    p.add_argument("--crop_y", type=int, nargs=2, default=[45, 210])

    p.add_argument("--imagesTs_dir", type=str, default=None)
    p.add_argument("--labelsTs_dir", type=str, default=None)

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--nerve_root_label", type=int, default=1)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    imagesTs_dir = args.imagesTs_dir or os.path.join(args.dataset_root, "imagesTs")
    labelsTs_dir = args.labelsTs_dir or os.path.join(args.dataset_root, "labelsTs")
    if not os.path.isdir(imagesTs_dir) or not os.path.isdir(labelsTs_dir):
        raise FileNotFoundError("imagesTs_dir / labelsTs_dir not found")

    model = load_model_from_ckpt(
        args.ckpt, device=device, base_channels=args.base_channels
    )

    run_test(
        model=model,
        imagesTs_dir=imagesTs_dir,
        labelsTs_dir=labelsTs_dir,
        out_dir=args.out_dir,
        device=device,
        crop_x=tuple(args.crop_x),
        crop_y=tuple(args.crop_y),
        nerve_root_label=args.nerve_root_label,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
