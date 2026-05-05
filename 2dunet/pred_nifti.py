#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn

try:
    import scipy.ndimage as ndi
except ImportError as e:
    raise ImportError("scipy が必要です: pip install scipy") from e


# ============================
# Model (must match train_all.py)
# ============================


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        self.up1_root = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)
        self.dec1_root = DoubleConv3D(base_channels * 2, base_channels)
        self.out_root = nn.Conv3d(base_channels, 1, 1)

        self.up1_dura = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)
        self.dec1_dura = DoubleConv3D(base_channels * 2, base_channels)
        self.out_dura = nn.Conv3d(base_channels, 1, 1)

    def _center_crop_to(self, enc: torch.Tensor, ref: torch.Tensor):
        _, _, d_ref, h_ref, w_ref = ref.size()
        _, _, d_enc, h_enc, w_enc = enc.size()
        d_t, h_t, w_t = min(d_ref, d_enc), min(h_ref, h_enc), min(w_ref, w_enc)
        d_s, h_s, w_s = (d_enc - d_t) // 2, (h_enc - h_t) // 2, (w_enc - w_t) // 2
        enc_c = enc[:, :, d_s : d_s + d_t, h_s : h_s + h_t, w_s : w_s + w_t]
        if (d_ref, h_ref, w_ref) != (d_t, h_t, w_t):
            d_sr, h_sr, w_sr = (
                (d_ref - d_t) // 2,
                (h_ref - h_t) // 2,
                (w_ref - w_t) // 2,
            )
            ref = ref[:, :, d_sr : d_sr + d_t, h_sr : h_sr + h_t, w_sr : w_sr + w_t]
        return enc_c, ref

    def forward(self, x):
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
# Utils
# ============================


def zscore_norm(img_zyx: np.ndarray) -> np.ndarray:
    m = float(img_zyx.mean())
    s = float(img_zyx.std())
    return (img_zyx.astype(np.float32) - m) / (s + 1e-6)


def minmax_norm(img_zyx: np.ndarray) -> np.ndarray:
    vmin = float(img_zyx.min())
    vmax = float(img_zyx.max())
    return (img_zyx.astype(np.float32) - vmin) / (vmax - vmin + 1e-6)


def resample_img_xyz(img_xyz, orig_spacing_xyz, target_spacing_xyz):
    sx, sy, sz = orig_spacing_xyz
    tx, ty, tz = target_spacing_xyz
    zoom = (sx / tx, sy / ty, sz / tz)
    return ndi.zoom(img_xyz, zoom=zoom, order=3).astype(np.float32)


def resample_mask_xyz_nearest(mask_xyz, orig_spacing_xyz, target_spacing_xyz):
    sx, sy, sz = orig_spacing_xyz
    tx, ty, tz = target_spacing_xyz
    zoom = (sx / tx, sy / ty, sz / tz)
    return ndi.zoom(mask_xyz, zoom=zoom, order=0).astype(np.uint8)


def load_model(ckpt_path: Path, device: torch.device, base_channels: int = 16):
    model = MultiTaskUNet3D(in_channels=1, base_channels=base_channels).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def fit_to_shape_center(src: np.ndarray, tgt_shape: Tuple[int, int, int]) -> np.ndarray:
    tx, ty, tz = tgt_shape
    dst = np.zeros((tx, ty, tz), dtype=src.dtype)
    sx, sy, sz = src.shape
    sx0 = max(0, (sx - tx) // 2)
    sy0 = max(0, (sy - ty) // 2)
    sz0 = max(0, (sz - tz) // 2)
    dx0 = max(0, (tx - sx) // 2)
    dy0 = max(0, (ty - sy) // 2)
    dz0 = max(0, (tz - sz) // 2)
    xlen = min(sx, tx)
    ylen = min(sy, ty)
    zlen = min(sz, tz)
    dst[dx0 : dx0 + xlen, dy0 : dy0 + ylen, dz0 : dz0 + zlen] = src[
        sx0 : sx0 + xlen, sy0 : sy0 + ylen, sz0 : sz0 + zlen
    ]
    return dst


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_nii", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out_root_nii", required=True, type=Path)
    ap.add_argument("--out_dura_nii", required=True, type=Path)

    ap.add_argument("--base_channels", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--crop_x", type=int, nargs=2, default=[50, 200])
    ap.add_argument("--crop_y", type=int, nargs=2, default=[45, 210])

    ap.add_argument("--enable_isotropic", action="store_true")
    ap.add_argument("--target_spacing_mm", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--enable_zscore_norm", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = load_model(args.ckpt, device=device, base_channels=args.base_channels)

    img_nii = nib.load(str(args.image_nii))
    img_xyz = img_nii.get_fdata().astype(np.float32)
    affine = img_nii.affine
    zooms = img_nii.header.get_zooms()[:3]
    orig_spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    x0, x1 = args.crop_x
    y0, y1 = args.crop_y
    img_crop_xyz = img_xyz[x0:x1, y0:y1, :]

    # inference grid
    if args.enable_isotropic:
        img_in_xyz = resample_img_xyz(
            img_crop_xyz, orig_spacing_xyz, tuple(args.target_spacing_mm)
        )
    else:
        img_in_xyz = img_crop_xyz

    # to ZYX
    img_in_zyx = np.transpose(img_in_xyz, (2, 1, 0))
    img_in_zyx = (
        zscore_norm(img_in_zyx) if args.enable_zscore_norm else minmax_norm(img_in_zyx)
    )

    x = torch.from_numpy(img_in_zyx[None, None, ...]).to(device)
    logits_root, logits_dura = model(x)

    prob_root = torch.sigmoid(logits_root)[0, 0].cpu().numpy()
    prob_dura = torch.sigmoid(logits_dura)[0, 0].cpu().numpy()

    pred_root_zyx = (prob_root > args.threshold).astype(np.uint8)
    pred_dura_zyx = (prob_dura > args.threshold).astype(np.uint8)

    # back to XYZ on inference grid
    pred_root_xyz_in = np.transpose(pred_root_zyx, (2, 1, 0))
    pred_dura_xyz_in = np.transpose(pred_dura_zyx, (2, 1, 0))

    # back to cropped original grid size
    if args.enable_isotropic:
        root_back = resample_mask_xyz_nearest(
            pred_root_xyz_in,
            orig_spacing_xyz=tuple(args.target_spacing_mm),
            target_spacing_xyz=orig_spacing_xyz,
        )
        dura_back = resample_mask_xyz_nearest(
            pred_dura_xyz_in,
            orig_spacing_xyz=tuple(args.target_spacing_mm),
            target_spacing_xyz=orig_spacing_xyz,
        )
        pred_root_xyz_crop = fit_to_shape_center(root_back, img_crop_xyz.shape)
        pred_dura_xyz_crop = fit_to_shape_center(dura_back, img_crop_xyz.shape)
    else:
        pred_root_xyz_crop = fit_to_shape_center(pred_root_xyz_in, img_crop_xyz.shape)
        pred_dura_xyz_crop = fit_to_shape_center(pred_dura_xyz_in, img_crop_xyz.shape)

    # paste to full image shape
    full_root = np.zeros_like(img_xyz, dtype=np.uint8)
    full_dura = np.zeros_like(img_xyz, dtype=np.uint8)
    full_root[x0:x1, y0:y1, :] = pred_root_xyz_crop
    full_dura[x0:x1, y0:y1, :] = pred_dura_xyz_crop

    args.out_root_nii.parent.mkdir(parents=True, exist_ok=True)
    args.out_dura_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(full_root, affine), str(args.out_root_nii))
    nib.save(nib.Nifti1Image(full_dura, affine), str(args.out_dura_nii))

    print("Saved root:", args.out_root_nii, "shape:", full_root.shape)
    print("Saved dura:", args.out_dura_nii, "shape:", full_dura.shape)


if __name__ == "__main__":
    main()
