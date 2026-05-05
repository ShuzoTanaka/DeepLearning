# test_2d_unet_axial_multitask.py
# 学習済み 2D U-Net（Axial / multi-task: root+dura）で imagesTs/labelsTs を case-wise 推論し、
# 3D再構成して metrics を計算 → CSV 出力（＋任意で予測NIfTI保存）
#
# 使い方例:
#   python test_2d_unet_axial_multitask.py ^
#     --dataset_root Dataset ^
#     --checkpoint ./ckpt2d_mt/best_unet2d_mt.pth ^
#     --out_dir ./ckpt2d_mt/pred_test ^
#     --crop_x 50 200 --crop_y 45 210
#
import os
import argparse
import csv
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn

try:
    import scipy.ndimage as ndi
    from scipy.ndimage import distance_transform_edt, binary_erosion
except ImportError as e:
    raise ImportError("このスクリプトには scipy が必要です: pip install scipy") from e


# =========================================================
# I/O utilities
# =========================================================
def _list_nii_files(folder: str) -> List[str]:
    if not os.path.exists(folder):
        return []
    files = [
        f for f in os.listdir(folder) if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    files.sort()
    return files


def case_id_from_label_path(lab_path: str) -> str:
    base = os.path.basename(lab_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def pair_paths(dataset_root: str, split: str) -> Tuple[List[str], List[str]]:
    """
    split = "Tr" or "Ts"
    images{split}/case_0000.nii.gz
    labels{split}/case.nii.gz
    """
    img_dir = os.path.join(dataset_root, f"images{split}")
    lab_dir = os.path.join(dataset_root, f"labels{split}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"{img_dir} not found")
    if not os.path.exists(lab_dir):
        raise FileNotFoundError(f"{lab_dir} not found")

    label_files = _list_nii_files(lab_dir)
    if len(label_files) == 0:
        raise RuntimeError(f"No label files found in {lab_dir}")

    image_paths, label_paths = [], []
    for lf in label_files:
        cid = lf.replace(".nii.gz", "").replace(".nii", "")
        img_name = f"{cid}_0000.nii.gz"
        img_path = os.path.join(img_dir, img_name)
        lab_path = os.path.join(lab_dir, lf)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found for label {lf}: expected {img_path}"
            )
        image_paths.append(img_path)
        label_paths.append(lab_path)

    return image_paths, label_paths


# =========================================================
# 2D U-Net (multi-task)
# =========================================================
class DoubleConv2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2DMultiTask(nn.Module):
    """
    入力:  (B,1,H,W)
    出力:  logits_root (B,1,H,W), logits_dura (B,1,H,W)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.enc1 = DoubleConv2D(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv2D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv2D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv2D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv2D(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = DoubleConv2D(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv2D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv2D(base_channels * 4, base_channels * 2)

        self.up1_root = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1_root = DoubleConv2D(base_channels * 2, base_channels)
        self.out_root = nn.Conv2d(base_channels, 1, 1)

        self.up1_dura = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1_dura = DoubleConv2D(base_channels * 2, base_channels)
        self.out_dura = nn.Conv2d(base_channels, 1, 1)

    @staticmethod
    def _center_crop_2d(
        enc: torch.Tensor, ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h_ref, w_ref = ref.shape
        _, _, h_enc, w_enc = enc.shape
        h_t = min(h_ref, h_enc)
        w_t = min(w_ref, w_enc)

        hs = (h_enc - h_t) // 2
        ws = (w_enc - w_t) // 2
        enc_c = enc[:, :, hs : hs + h_t, ws : ws + w_t]

        if (h_ref, w_ref) != (h_t, w_t):
            hs2 = (h_ref - h_t) // 2
            ws2 = (w_ref - w_t) // 2
            ref = ref[:, :, hs2 : hs2 + h_t, ws2 : ws2 + w_t]
        return enc_c, ref

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        e4c, u4 = self._center_crop_2d(e4, u4)
        d4 = self.dec4(torch.cat([u4, e4c], dim=1))

        u3 = self.up3(d4)
        e3c, u3 = self._center_crop_2d(e3, u3)
        d3 = self.dec3(torch.cat([u3, e3c], dim=1))

        u2 = self.up2(d3)
        e2c, u2 = self._center_crop_2d(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2c], dim=1))

        u1r = self.up1_root(d2)
        e1cr, u1r = self._center_crop_2d(e1, u1r)
        d1r = self.dec1_root(torch.cat([u1r, e1cr], dim=1))
        out_root = self.out_root(d1r)

        u1d = self.up1_dura(d2)
        e1cd, u1d = self._center_crop_2d(e1, u1d)
        d1d = self.dec1_dura(torch.cat([u1d, e1cd], dim=1))
        out_dura = self.out_dura(d1d)

        return out_root, out_dura


# =========================================================
# Metrics (3D)
# =========================================================
def boundary_mask_3d(binmask: np.ndarray) -> np.ndarray:
    binmask = (binmask > 0).astype(bool)
    if binmask.sum() == 0:
        return np.zeros_like(binmask, dtype=bool)
    er = binary_erosion(binmask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    return binmask & (~er)


def dice_coeff_3d(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum()) + eps
    return 2.0 * inter / denom


def surface_distances_mm(
    a: np.ndarray, b: np.ndarray, spacing_xyz: Tuple[float, float, float]
) -> np.ndarray:
    a_bd = boundary_mask_3d(a)
    b = (b > 0).astype(bool)
    if a_bd.sum() == 0:
        return np.array([], dtype=np.float32)
    if b.sum() == 0:
        return np.array([np.inf], dtype=np.float32)
    dt = distance_transform_edt(~b, sampling=spacing_xyz)
    return dt[a_bd].astype(np.float32)


def hd95_asd_mm(
    pred: np.ndarray, gt: np.ndarray, spacing_xyz: Tuple[float, float, float]
) -> Tuple[float, float]:
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf"), float("inf")

    d1 = surface_distances_mm(pred, gt, spacing_xyz)
    d2 = surface_distances_mm(gt, pred, spacing_xyz)
    d = np.concatenate([d1, d2], axis=0)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("inf"), float("inf")
    return float(np.percentile(d, 95)), float(d.mean())


def boundary_iou_3d(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pb = boundary_mask_3d(pred > 0)
    gb = boundary_mask_3d(gt > 0)
    inter = float((pb & gb).sum())
    union = float((pb | gb).sum()) + eps
    return inter / union


# =========================================================
# Test (case-wise): slice inference -> reconstruct 3D -> metrics -> CSV
# =========================================================
@torch.no_grad()
def run_test_casewise(
    model: UNet2DMultiTask,
    ts_imgs: List[str],
    ts_labs: List[str],
    out_dir: str,
    root_label: int,
    dura_label: int,
    thr_root: float,
    thr_dura: float,
    crop_x: Optional[Tuple[int, int]],
    crop_y: Optional[Tuple[int, int]],
    save_nifti: bool,
) -> None:
    device = next(model.parameters()).device
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, float]] = []

    for img_path, lab_path in tqdm(
        list(zip(ts_imgs, ts_labs)), desc="Test", leave=False
    ):
        cid = case_id_from_label_path(lab_path)

        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        img3d = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)
        lab3d = lab_nii.get_fdata().astype(np.int16)

        if img3d.shape != lab3d.shape:
            raise ValueError(
                f"Shape mismatch in test {cid}: img{img3d.shape} vs lab{lab3d.shape}"
            )

        # optional crop in X,Y
        if crop_x is not None and crop_y is not None:
            x0, x1 = crop_x
            y0, y1 = crop_y
            img3d = img3d[x0:x1, y0:y1, :]
            lab3d = lab3d[x0:x1, y0:y1, :]

        # normalize per-volume
        vmin, vmax = float(img3d.min()), float(img3d.max())
        if vmax > vmin:
            img3d = (img3d - vmin) / (vmax - vmin)
        else:
            img3d = np.zeros_like(img3d, dtype=np.float32)

        zooms = img_nii.header.get_zooms()[:3]
        spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        X, Y, Z = img3d.shape
        pr_root = np.zeros((X, Y, Z), dtype=np.uint8)
        pr_dura = np.zeros((X, Y, Z), dtype=np.uint8)

        for z in range(Z):
            sl = img3d[:, :, z].astype(np.float32)
            inp = torch.from_numpy(sl[None, None, ...]).to(device)

            log_r, log_d = model(inp)
            p_r = (torch.sigmoid(log_r)[0, 0].cpu().numpy() > thr_root).astype(np.uint8)
            p_d = (torch.sigmoid(log_d)[0, 0].cpu().numpy() > thr_dura).astype(np.uint8)

            # safety: center crop/pad to match slice size
            hx, wx = p_r.shape
            tx, ty = sl.shape
            if (hx, wx) != (tx, ty):
                out_r = np.zeros((tx, ty), dtype=np.uint8)
                out_d = np.zeros((tx, ty), dtype=np.uint8)

                hs = max((hx - tx) // 2, 0)
                ws = max((wx - ty) // 2, 0)
                hd = max((tx - hx) // 2, 0)
                wd = max((ty - wx) // 2, 0)

                cx = min(hx, tx)
                cy = min(wx, ty)

                out_r[hd : hd + cx, wd : wd + cy] = p_r[hs : hs + cx, ws : ws + cy]
                out_d[hd : hd + cx, wd : wd + cy] = p_d[hs : hs + cx, ws : ws + cy]
                p_r, p_d = out_r, out_d

            pr_root[:, :, z] = p_r
            pr_dura[:, :, z] = p_d

        gt_root = (lab3d == root_label).astype(np.uint8)
        gt_dura = (lab3d == dura_label).astype(np.uint8)

        root_dice = dice_coeff_3d(pr_root, gt_root)
        root_hd95, root_asd = hd95_asd_mm(pr_root, gt_root, spacing_xyz)
        root_biou = boundary_iou_3d(pr_root, gt_root)

        dura_dice = dice_coeff_3d(pr_dura, gt_dura)
        dura_hd95, dura_asd = hd95_asd_mm(pr_dura, gt_dura, spacing_xyz)
        dura_biou = boundary_iou_3d(pr_dura, gt_dura)

        rows.append(
            {
                "case_id": cid,
                "root_dice": root_dice,
                "root_hd95_mm": root_hd95,
                "root_asd_mm": root_asd,
                "root_boundary_iou": root_biou,
                "dura_dice": dura_dice,
                "dura_hd95_mm": dura_hd95,
                "dura_asd_mm": dura_asd,
                "dura_boundary_iou": dura_biou,
            }
        )

        print(
            f"\nCase {cid} | "
            f"root Dice={root_dice:.4f}, HD95={root_hd95:.3f}mm, ASD={root_asd:.3f}mm, bIoU={root_biou:.4f} | "
            f"dura Dice={dura_dice:.4f}, HD95={dura_hd95:.3f}mm, ASD={dura_asd:.3f}mm, bIoU={dura_biou:.4f}"
        )

        if save_nifti:
            if crop_x is not None and crop_y is not None:
                full_shape = img_nii.shape
                root_full = np.zeros(full_shape, dtype=np.uint8)
                dura_full = np.zeros(full_shape, dtype=np.uint8)
                x0, x1 = crop_x
                y0, y1 = crop_y
                root_full[x0:x1, y0:y1, :] = pr_root
                dura_full[x0:x1, y0:y1, :] = pr_dura
            else:
                root_full = pr_root
                dura_full = pr_dura

            root_nii = nib.Nifti1Image(
                root_full.astype(np.uint8), img_nii.affine, img_nii.header
            )
            dura_nii = nib.Nifti1Image(
                dura_full.astype(np.uint8), img_nii.affine, img_nii.header
            )
            root_nii.set_data_dtype(np.uint8)
            dura_nii.set_data_dtype(np.uint8)
            nib.save(root_nii, os.path.join(out_dir, f"{cid}_root_pred.nii.gz"))
            nib.save(dura_nii, os.path.join(out_dir, f"{cid}_dura_pred.nii.gz"))

    # ===== CSV =====
    csv_path = os.path.join(out_dir, "metrics_test.csv")
    header = [
        "case_id",
        "root_dice",
        "root_hd95_mm",
        "root_asd_mm",
        "root_boundary_iou",
        "dura_dice",
        "dura_hd95_mm",
        "dura_asd_mm",
        "dura_boundary_iou",
    ]

    def fmt(x: float) -> str:
        return f"{x:.6f}" if np.isfinite(x) else "inf"

    def mean_finite(vals: List[float]) -> float:
        v = np.asarray(vals, dtype=np.float32)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size > 0 else float("nan")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(
                [
                    r["case_id"],
                    fmt(r["root_dice"]),
                    fmt(r["root_hd95_mm"]),
                    fmt(r["root_asd_mm"]),
                    fmt(r["root_boundary_iou"]),
                    fmt(r["dura_dice"]),
                    fmt(r["dura_hd95_mm"]),
                    fmt(r["dura_asd_mm"]),
                    fmt(r["dura_boundary_iou"]),
                ]
            )

        w.writerow(
            [
                "MEAN",
                fmt(float(np.mean([r["root_dice"] for r in rows]))),
                fmt(mean_finite([r["root_hd95_mm"] for r in rows])),
                fmt(mean_finite([r["root_asd_mm"] for r in rows])),
                fmt(float(np.mean([r["root_boundary_iou"] for r in rows]))),
                fmt(float(np.mean([r["dura_dice"] for r in rows]))),
                fmt(mean_finite([r["dura_hd95_mm"] for r in rows])),
                fmt(mean_finite([r["dura_asd_mm"] for r in rows])),
                fmt(float(np.mean([r["dura_boundary_iou"] for r in rows]))),
            ]
        )

    print(f"\nSaved test metrics CSV: {csv_path}")


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser("Test 2D U-Net axial multi-task (root+dura) -> CSV")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--root_label", type=int, default=1)
    ap.add_argument("--dura_label", type=int, default=2)

    ap.add_argument("--thr_root", type=float, default=0.5)
    ap.add_argument("--thr_dura", type=float, default=0.5)

    ap.add_argument("--base_channels", type=int, default=32)

    ap.add_argument("--crop_x", type=int, nargs=2, default=None)
    ap.add_argument("--crop_y", type=int, nargs=2, default=None)

    ap.add_argument("--save_nifti", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- load model ----
    model = UNet2DMultiTask(in_channels=1, base_channels=args.base_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    model.eval()
    print("Loaded checkpoint:", args.checkpoint)

    # ---- test pairs ----
    ts_imgs, ts_labs = pair_paths(args.dataset_root, "Ts")
    print(f"#Test cases: {len(ts_imgs)}")

    crop_x = tuple(args.crop_x) if args.crop_x is not None else None
    crop_y = tuple(args.crop_y) if args.crop_y is not None else None

    run_test_casewise(
        model=model,
        ts_imgs=ts_imgs,
        ts_labs=ts_labs,
        out_dir=args.out_dir,
        root_label=args.root_label,
        dura_label=args.dura_label,
        thr_root=args.thr_root,
        thr_dura=args.thr_dura,
        crop_x=crop_x,
        crop_y=crop_y,
        save_nifti=bool(args.save_nifti),
    )


if __name__ == "__main__":
    main()
