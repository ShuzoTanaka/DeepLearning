#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-case metrics between GT and PRED NIfTI masks:
case_id,dice,hd95_mm,asd_mm,boundary_iou
+ final MEAN row.

- Uses NIfTI spacing (mm) from header.get_zooms()[:3]
- Evaluates binary foreground: (label > 0)
- Requires: numpy, nibabel, scipy
"""

from pathlib import Path
import argparse
import csv
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt


def load_mask(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} for {path}")
    spacing = img.header.get_zooms()[:3]  # (sx, sy, sz) in mm
    return data, spacing


def dice_coef(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    sa = a.sum(dtype=np.float64)
    sb = b.sum(dtype=np.float64)
    return float((2.0 * inter + eps) / (sa + sb + eps))


def surface_mask(mask: np.ndarray) -> np.ndarray:
    """1-voxel thick boundary of a binary mask"""
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    er = binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(er))


def surface_distances_mm(gt: np.ndarray, pr: np.ndarray, spacing) -> np.ndarray | None:
    """
    Compute symmetric surface distances in mm using distance transforms.
    Returns concatenated distances from GT->PR and PR->GT.
    """
    gt_s = surface_mask(gt)
    pr_s = surface_mask(pr)

    # If either surface is empty, distances are undefined
    if gt_s.sum() == 0 or pr_s.sum() == 0:
        return None

    # Distance to nearest surface voxel (sampling uses spacing => mm)
    dt_pr = distance_transform_edt(~pr_s, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_s, sampling=spacing)

    d_gt_to_pr = dt_pr[gt_s]
    d_pr_to_gt = dt_gt[pr_s]
    return np.concatenate([d_gt_to_pr, d_pr_to_gt]).astype(np.float64)


def hd95_asd_mm(gt: np.ndarray, pr: np.ndarray, spacing):
    d = surface_distances_mm(gt, pr, spacing)
    if d is None or d.size == 0:
        return np.nan, np.nan
    hd95 = float(np.percentile(d, 95))
    asd = float(d.mean())
    return hd95, asd


def boundary_iou(gt: np.ndarray, pr: np.ndarray) -> float:
    gt_b = surface_mask(gt)
    pr_b = surface_mask(pr)
    union = np.logical_or(gt_b, pr_b).sum(dtype=np.float64)
    inter = np.logical_and(gt_b, pr_b).sum(dtype=np.float64)
    if union == 0:
        # both empty boundaries => treat as perfect
        return 1.0
    return float(inter / union)


def mean_ignore_nan(rows, key: str):
    vals = [r[key] for r in rows if r[key] is not None and not np.isnan(r[key])]
    return float(np.mean(vals)) if len(vals) > 0 else np.nan


def find_pred_file(pred_dir: Path, gt_path: Path) -> Path | None:
    """
    Match prediction file to GT filename.
    Priority:
    1) same filename
    2) same stem prefix
    """
    cand = pred_dir / gt_path.name
    if cand.exists():
        return cand

    case_id = gt_path.stem.replace(".nii", "")
    candidates = list(pred_dir.glob(case_id + "*"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path, help="labelsTs folder (GT)")
    ap.add_argument(
        "--pred", required=True, type=Path, help="pred_out_test folder (PRED)"
    )
    ap.add_argument("--out", required=True, type=Path, help="output csv path")
    ap.add_argument("--suffix", default=".nii.gz", help="file suffix, default .nii.gz")
    args = ap.parse_args()

    gt_dir: Path = args.gt
    pr_dir: Path = args.pred
    out_csv: Path = args.out

    gt_files = sorted(gt_dir.glob(f"*{args.suffix}"))
    if len(gt_files) == 0:
        raise RuntimeError(f"No GT files found in {gt_dir} with suffix {args.suffix}")

    rows = []
    missing_pred = []

    for gt_path in gt_files:
        case_id = gt_path.stem.replace(".nii", "")  # handles .nii.gz

        pr_path = find_pred_file(pr_dir, gt_path)
        if pr_path is None or (not pr_path.exists()):
            missing_pred.append(gt_path.name)
            continue

        gt_lab, spacing_gt = load_mask(gt_path)
        pr_lab, spacing_pr = load_mask(pr_path)

        if gt_lab.shape != pr_lab.shape:
            raise ValueError(
                f"Shape mismatch: {gt_path.name} {gt_lab.shape} vs {pr_path.name} {pr_lab.shape}"
            )

        # Use GT spacing for mm metrics (safe & standard)
        spacing_mm = spacing_gt

        # Evaluate foreground as binary
        gt_fg = gt_lab > 0
        pr_fg = pr_lab > 0

        dice = dice_coef(gt_fg, pr_fg)
        hd95, asd = hd95_asd_mm(gt_fg, pr_fg, spacing_mm)
        biou = boundary_iou(gt_fg, pr_fg)

        rows.append(
            {
                "case_id": case_id,
                "dice": dice,
                "hd95_mm": hd95,
                "asd_mm": asd,
                "boundary_iou": biou,
            }
        )

    # --- MEAN row (ignore NaN) ---
    mean_row = {
        "case_id": "MEAN",
        "dice": mean_ignore_nan(rows, "dice"),
        "hd95_mm": mean_ignore_nan(rows, "hd95_mm"),
        "asd_mm": mean_ignore_nan(rows, "asd_mm"),
        "boundary_iou": mean_ignore_nan(rows, "boundary_iou"),
    }

    # --- write CSV ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["case_id", "dice", "hd95_mm", "asd_mm", "boundary_iou"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow(mean_row)

    print(f"[OK] wrote: {out_csv}  (n={len(rows)})")
    if missing_pred:
        print(
            f"[WARN] missing predictions for {len(missing_pred)} cases, examples: {missing_pred[:5]}"
        )


if __name__ == "__main__":
    main()
