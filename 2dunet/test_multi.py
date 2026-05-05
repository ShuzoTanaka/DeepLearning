# test_mt.py
import os
import argparse
import csv
from typing import Dict, Tuple, List

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch

# ==== HD95/ASD 用（scipyが必要）====
try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
except ImportError as e:
    raise ImportError("HD95/ASD計算に scipy が必要です: pip install scipy") from e


# =========================
# ここはあなたの train.py からコピペ or import
# =========================
# もし train スクリプト名が train_3d_mt.py 等なら、その名前に合わせて import してください。
from train_3d_multi import (
    MultiTaskUNet3D,
    pair_ts_paths,
)  # ← trainファイル名に合わせて変更！


# =========================
# Fixed crop (学習と同じ)
# =========================
CROP_X = (50, 200)  # x: 50..199
CROP_Y = (45, 210)  # y: 45..209


def center_crop_3d_to_match_np(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    a, b: (X,Y,Z) 3D
    → (minX, minY, minZ) に中心クロップしてサイズを揃える
    """
    assert a.ndim == 3 and b.ndim == 3
    xa, ya, za = a.shape
    xb, yb, zb = b.shape

    xt = min(xa, xb)
    yt = min(ya, yb)
    zt = min(za, zb)

    def crop(v, xt, yt, zt):
        x, y, z = v.shape
        xs = (x - xt) // 2
        ys = (y - yt) // 2
        zs = (z - zt) // 2
        return v[xs : xs + xt, ys : ys + yt, zs : zs + zt]

    return crop(a, xt, yt, zt), crop(b, xt, yt, zt)


def case_id_from_label_path(lab_path: str) -> str:
    base = os.path.basename(lab_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def load_and_preprocess_case(
    img_path: str,
    lab_path: str,
    root_label: int,
    dura_label: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Image, Tuple[float, float, float]
]:
    """
    返り値:
      img_crop_norm: (Xc,Yc,Z) float32 0-1
      gt_root:       (Xc,Yc,Z) uint8(0/1)
      gt_dura:       (Xc,Yc,Z) uint8(0/1)
      img_nii: nib object (affine/header用)
      spacing: (sx,sy,sz) mm
    """
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)

    lab_nii = nib.load(lab_path)
    lab = lab_nii.get_fdata().astype(np.int16)

    if img.shape != lab.shape:
        raise ValueError(f"Shape mismatch: img {img.shape} vs lab {lab.shape}")

    # spacing (mm)
    zooms = img_nii.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    # crop (学習と同じ固定範囲)
    x0, x1 = CROP_X
    y0, y1 = CROP_Y
    img_c = img[x0:x1, y0:y1, :]
    lab_c = lab[x0:x1, y0:y1, :]

    # normalize 0-1 (crop後)
    vmin, vmax = float(img_c.min()), float(img_c.max())
    if vmax > vmin:
        img_c = (img_c - vmin) / (vmax - vmin)
    else:
        img_c = np.zeros_like(img_c, dtype=np.float32)

    gt_root = (lab_c == root_label).astype(np.uint8)
    gt_dura = (lab_c == dura_label).astype(np.uint8)

    return img_c.astype(np.float32), gt_root, gt_dura, img_nii, spacing


def dice_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred, gt = center_crop_3d_to_match_np(pred, gt)  # ★追加

    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum()) + eps
    return 2.0 * inter / denom


def boundary_mask(binmask: np.ndarray) -> np.ndarray:
    """
    3D境界: mask - erosion(mask)
    """
    binmask = (binmask > 0).astype(bool)
    if binmask.sum() == 0:
        return np.zeros_like(binmask, dtype=bool)
    # 3x3x3
    er = binary_erosion(binmask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    bd = binmask & (~er)
    return bd


def surface_distances_mm(
    a: np.ndarray, b: np.ndarray, spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    a,b: bool 3D
    a境界→b への距離 (mm) を返す
    """
    a_bd = boundary_mask(a)
    b = (b > 0).astype(bool)

    if a_bd.sum() == 0:
        return np.array([], dtype=np.float32)
    if b.sum() == 0:
        # 相手が空なら距離は無限大相当（ここでは大きい値にして扱いを分離してもよいが、infにする）
        return np.array([np.inf], dtype=np.float32)

    # bの補集合のdistance transform（bの最近傍までの距離）
    dt = distance_transform_edt(~b, sampling=spacing)  # mm
    d = dt[a_bd]
    return d.astype(np.float32)


def hd95_asd_mm(
    pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[float, float]:
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    pred, gt = center_crop_3d_to_match_np(pred.astype(np.uint8), gt.astype(np.uint8))
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf"), float("inf")

    d1 = surface_distances_mm(pred, gt, spacing)
    d2 = surface_distances_mm(gt, pred, spacing)
    d = np.concatenate([d1, d2], axis=0)

    d_f = d[np.isfinite(d)]
    if d_f.size == 0:
        return float("inf"), float("inf")

    hd95 = float(np.percentile(d_f, 95))
    asd = float(d_f.mean())
    return hd95, asd


def boundary_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred, gt = center_crop_3d_to_match_np(pred, gt)  # ★追加

    pb = boundary_mask(pred > 0)
    gb = boundary_mask(gt > 0)
    inter = float((pb & gb).sum())
    union = float((pb | gb).sum()) + eps
    return inter / union


def embed_crop_back_to_full(
    pred_crop: np.ndarray, full_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    pred_crop: (Xc,Yc,Zc)  ←モデル出力の2値マスク（cropped空間）
    full_shape: (X,Y,Z)    ←元画像のshape
    戻り値: (X,Y,Z) に中心埋め(0埋め)

    ※ ここでは “固定 crop 座標(50:200,45:210)” ではなく、
       pred_crop のサイズに合わせて中央に戻します。
    """
    X, Y, Z = full_shape
    xc, yc, zc = pred_crop.shape

    out = np.zeros((X, Y, Z), dtype=pred_crop.dtype)

    if xc > X or yc > Y or zc > Z:
        raise ValueError(
            f"pred_crop {pred_crop.shape} is larger than full {full_shape}"
        )

    x0 = (X - xc) // 2
    y0 = (Y - yc) // 2
    z0 = (Z - zc) // 2

    out[x0 : x0 + xc, y0 : y0 + yc, z0 : z0 + zc] = pred_crop
    return out


def run_test(
    dataset_root: str,
    checkpoint_path: str,
    out_dir: str,
    root_label: int = 1,
    dura_label: int = 2,
    thr_root: float = 0.5,
    thr_dura: float = 0.5,
    save_nifti: bool = True,
) -> Dict[str, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ts_imgs, ts_labs = pair_ts_paths(dataset_root)
    print(f"#Test cases: {len(ts_imgs)}")

    os.makedirs(out_dir, exist_ok=True)

    model = MultiTaskUNet3D(in_channels=1, base_channels=16).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    )
    model.eval()

    metrics_by_case: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for img_path, lab_path in tqdm(
            list(zip(ts_imgs, ts_labs)), desc="Test", leave=False
        ):
            cid = case_id_from_label_path(lab_path)

            img_c, gt_root, gt_dura, img_nii, spacing = load_and_preprocess_case(
                img_path, lab_path, root_label, dura_label
            )

            # (Xc,Yc,Z) -> torch (1,1,Xc,Yc,Z)
            vol = torch.from_numpy(img_c[None, ...]).unsqueeze(0).to(device)

            logits_root, logits_dura = model(vol)

            pr_root = (
                (torch.sigmoid(logits_root) > thr_root)
                .float()
                .cpu()
                .numpy()[0, 0]
                .astype(np.uint8)
            )
            pr_dura = (
                (torch.sigmoid(logits_dura) > thr_dura)
                .float()
                .cpu()
                .numpy()[0, 0]
                .astype(np.uint8)
            )

            # metrics (root)
            root_dice = dice_coeff(pr_root, gt_root)
            root_hd95, root_asd = hd95_asd_mm(pr_root, gt_root, spacing)
            root_biou = boundary_iou(pr_root, gt_root)

            # metrics (dura)
            dura_dice = dice_coeff(pr_dura, gt_dura)
            dura_hd95, dura_asd = hd95_asd_mm(pr_dura, gt_dura, spacing)
            dura_biou = boundary_iou(pr_dura, gt_dura)

            metrics_by_case[cid] = {
                "root_dice": float(root_dice),
                "root_hd95": float(root_hd95),
                "root_asd": float(root_asd),
                "root_biou": float(root_biou),
                "dura_dice": float(dura_dice),
                "dura_hd95": float(dura_hd95),
                "dura_asd": float(dura_asd),
                "dura_biou": float(dura_biou),
            }

            print(
                f"\nCase {cid} | "
                f"root Dice={root_dice:.4f}, HD95={root_hd95:.3f}, ASD={root_asd:.3f}, bIoU={root_biou:.4f} | "
                f"dura Dice={dura_dice:.4f}, HD95={dura_hd95:.3f}, ASD={dura_asd:.3f}, bIoU={dura_biou:.4f}"
            )

            # save nifti
            if save_nifti:
                full_shape = img_nii.shape  # (X,Y,Z)
                root_full = embed_crop_back_to_full(pr_root, full_shape)
                dura_full = embed_crop_back_to_full(pr_dura, full_shape)

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
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
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
        )

        for cid, m in metrics_by_case.items():

            def fmt(x):
                return f"{x:.6f}" if np.isfinite(x) else "inf"

            w.writerow(
                [
                    cid,
                    f"{m['root_dice']:.6f}",
                    fmt(m["root_hd95"]),
                    fmt(m["root_asd"]),
                    f"{m['root_biou']:.6f}",
                    f"{m['dura_dice']:.6f}",
                    fmt(m["dura_hd95"]),
                    fmt(m["dura_asd"]),
                    f"{m['dura_biou']:.6f}",
                ]
            )

        def mean_finite(vals):
            v = np.asarray(vals, dtype=np.float32)
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size > 0 else float("nan")

        root_dice_mean = float(
            np.mean([m["root_dice"] for m in metrics_by_case.values()])
        )
        root_hd95_mean = mean_finite([m["root_hd95"] for m in metrics_by_case.values()])
        root_asd_mean = mean_finite([m["root_asd"] for m in metrics_by_case.values()])
        root_biou_mean = float(
            np.mean([m["root_biou"] for m in metrics_by_case.values()])
        )

        dura_dice_mean = float(
            np.mean([m["dura_dice"] for m in metrics_by_case.values()])
        )
        dura_hd95_mean = mean_finite([m["dura_hd95"] for m in metrics_by_case.values()])
        dura_asd_mean = mean_finite([m["dura_asd"] for m in metrics_by_case.values()])
        dura_biou_mean = float(
            np.mean([m["dura_biou"] for m in metrics_by_case.values()])
        )

        def fmt(x):
            return f"{x:.6f}" if np.isfinite(x) else "inf"

        w.writerow(
            [
                "MEAN",
                f"{root_dice_mean:.6f}",
                fmt(root_hd95_mean),
                fmt(root_asd_mean),
                f"{root_biou_mean:.6f}",
                f"{dura_dice_mean:.6f}",
                fmt(dura_hd95_mean),
                fmt(dura_asd_mean),
                f"{dura_biou_mean:.6f}",
            ]
        )

    print(f"\nSaved metrics CSV: {csv_path}")
    return metrics_by_case


def main():
    p = argparse.ArgumentParser(
        "MultiTask 3D U-Net test (root + dura) with metrics + CSV"
    )
    p.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset (imagesTs/labelsTs)"
    )
    p.add_argument("--checkpoint", type=str, required=True, help="trainで保存した .pth")
    p.add_argument("--out_dir", type=str, default="./pred3d_mt", help="出力フォルダ")
    p.add_argument("--root_label", type=int, default=1)
    p.add_argument("--dura_label", type=int, default=2)
    p.add_argument("--thr_root", type=float, default=0.5)
    p.add_argument("--thr_dura", type=float, default=0.5)
    p.add_argument("--no_save_nifti", action="store_true", help="予測NIfTIを保存しない")
    args = p.parse_args()

    run_test(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        root_label=args.root_label,
        dura_label=args.dura_label,
        thr_root=args.thr_root,
        thr_dura=args.thr_dura,
        save_nifti=not args.no_save_nifti,
    )


if __name__ == "__main__":
    main()
