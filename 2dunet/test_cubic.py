import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import scipy.ndimage as ndi

from train_3d import UNet3D, pair_ts_paths, dice_coeff_numpy


# ----------------------------
# Resample (same as train)
# ----------------------------
def resample_to_isotropic(
    img: np.ndarray,
    lab: np.ndarray | None,
    orig_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    sx, sy, sz = orig_spacing
    tx, ty, tz = target_spacing
    zoom_factors = (sx / tx, sy / ty, sz / tz)

    img_iso = ndi.zoom(img, zoom=zoom_factors, order=3).astype(np.float32)

    if lab is not None:
        lab_iso = ndi.zoom(lab, zoom=zoom_factors, order=0).astype(np.int16)
        return img_iso, lab_iso

    return img_iso, None


# ----------------------------
# Metrics (surface distances)
# ----------------------------
def _surface(mask: np.ndarray) -> np.ndarray:
    """1-voxel surface: mask XOR erode(mask)"""
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    er = ndi.binary_erosion(mask, iterations=1)
    return mask ^ er


def hd95_mm(
    pred: np.ndarray, gt: np.ndarray, spacing_mm: Tuple[float, float, float]
) -> float:
    """
    Robust Hausdorff (95th percentile) in mm.
    If one is empty and the other is not -> inf.
    If both empty -> 0.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    sp = np.array(spacing_mm, dtype=np.float32)

    pred_s = _surface(pred)
    gt_s = _surface(gt)

    # distance to the other surface: use EDT of inverse(surface)
    dt_gt = ndi.distance_transform_edt(~gt_s, sampling=sp)
    dt_pred = ndi.distance_transform_edt(~pred_s, sampling=sp)

    d_pred_to_gt = dt_gt[pred_s]
    d_gt_to_pred = dt_pred[gt_s]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float32)
    if all_d.size == 0:
        return 0.0
    return float(np.percentile(all_d, 95))


def asd_mm(
    pred: np.ndarray, gt: np.ndarray, spacing_mm: Tuple[float, float, float]
) -> float:
    """
    Average Symmetric Surface Distance in mm.
    same empty handling as hd95.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    sp = np.array(spacing_mm, dtype=np.float32)

    pred_s = _surface(pred)
    gt_s = _surface(gt)

    dt_gt = ndi.distance_transform_edt(~gt_s, sampling=sp)
    dt_pred = ndi.distance_transform_edt(~pred_s, sampling=sp)

    d_pred_to_gt = dt_gt[pred_s]
    d_gt_to_pred = dt_pred[gt_s]

    if d_pred_to_gt.size == 0 and d_gt_to_pred.size == 0:
        return 0.0

    return float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0)


def boundary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Boundary IoU on 1-voxel surfaces (no dilation).
    If both boundaries empty -> 1.
    If union=0 -> 1 (same as both empty).
    """
    pred_s = _surface(pred.astype(bool))
    gt_s = _surface(gt.astype(bool))

    inter = np.logical_and(pred_s, gt_s).sum()
    union = np.logical_or(pred_s, gt_s).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


# ----------------------------
# CSV
# ----------------------------
def save_csv_4cols(rows: List[Dict], csv_path: str):
    import csv

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = ["case_id", "dice", "hd95_mm", "asd_mm", "boundary_iou"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def case_id_from_label_path(lab_path: str) -> str:
    base = os.path.basename(lab_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


@torch.no_grad()
def run_test(
    dataset_root: str,
    checkpoint_path: str,
    out_dir: str,
    nerve_root_label: int = 1,
    threshold: float = 0.5,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ts_imgs, ts_labs = pair_ts_paths(dataset_root)
    print(f"#Test cases: {len(ts_imgs)}")

    os.makedirs(out_dir, exist_ok=True)

    model = UNet3D(in_channels=1, base_channels=16).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    rows: List[Dict] = []
    dices, hd95s, asds, bious = [], [], [], []

    for img_path, lab_path in tqdm(
        list(zip(ts_imgs, ts_labs)), desc="Test", leave=False
    ):
        cid = case_id_from_label_path(lab_path)

        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        img = img_nii.get_fdata().astype(np.float32)
        lab = lab_nii.get_fdata().astype(np.int16)
        if img.shape != lab.shape:
            raise ValueError(
                f"Shape mismatch: img {img.shape} vs lab {lab.shape} ({cid})"
            )

        orig_spacing = img_nii.header.get_zooms()[:3]
        img, lab = resample_to_isotropic(
            img, lab, orig_spacing, target_spacing=target_spacing
        )

        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        gt = (lab == nerve_root_label).astype(np.uint8)

        vol = torch.from_numpy(img[None, ...]).unsqueeze(0).to(device)  # (1,1,X,Y,Z)
        logits = model(vol)
        prob = torch.sigmoid(logits)
        pred = (prob > threshold).float().cpu().numpy()[0, 0].astype(np.uint8)

        # Dice (center-crop inside)
        d = float(dice_coeff_numpy(pred, gt))

        # For surface metrics, also align shapes (same policy as dice)
        pred_c, gt_c = pred, gt
        if pred.shape != gt.shape:
            # center crop to min (same idea as train_3d)
            d_a, h_a, w_a = pred.shape
            d_b, h_b, w_b = gt.shape
            d_t, h_t, w_t = min(d_a, d_b), min(h_a, h_b), min(w_a, w_b)

            def crop(v, d_t, h_t, w_t):
                d, h, w = v.shape
                ds = (d - d_t) // 2
                hs = (h - h_t) // 2
                ws = (w - w_t) // 2
                return v[ds : ds + d_t, hs : hs + h_t, ws : ws + w_t]

            pred_c = crop(pred, d_t, h_t, w_t)
            gt_c = crop(gt, d_t, h_t, w_t)

        h = float(hd95_mm(pred_c, gt_c, spacing_mm=target_spacing))
        a = float(asd_mm(pred_c, gt_c, spacing_mm=target_spacing))
        b = float(boundary_iou(pred_c, gt_c))

        rows.append(
            {
                "case_id": cid,
                "dice": d,
                "hd95_mm": h,
                "asd_mm": a,
                "boundary_iou": b,
            }
        )

        dices.append(d)
        hd95s.append(h)
        asds.append(a)
        bious.append(b)
        print(f"Case {cid} | Dice={d:.4f}, HD95={h:.3f}mm, ASD={a:.3f}mm, bIoU={b:.4f}")

    # mean row
    rows.append(
        {
            "case_id": "MEAN",
            "dice": float(np.mean(dices)) if dices else "",
            "hd95_mm": float(np.mean(hd95s)) if hd95s else "",
            "asd_mm": float(np.mean(asds)) if asds else "",
            "boundary_iou": float(np.mean(bious)) if bious else "",
        }
    )

    csv_path = os.path.join(out_dir, "test_metrics.csv")
    save_csv_4cols(rows, csv_path)
    print(f"\nSaved CSV: {csv_path}")


def main():
    p = argparse.ArgumentParser(
        description="3D U-Net test: Dice/HD95/ASD/BoundaryIoU -> CSV"
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./test_out")
    p.add_argument("--nerve_root_label", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target_spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    args = p.parse_args()

    run_test(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        nerve_root_label=args.nerve_root_label,
        threshold=args.threshold,
        target_spacing=tuple(args.target_spacing),
    )


if __name__ == "__main__":
    main()
