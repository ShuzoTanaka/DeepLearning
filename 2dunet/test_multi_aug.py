import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import scipy.ndimage as ndi

# ★ あなたの train ファイル名に合わせて import を調整してください
# 例: train_multi.py なら from train_multi import MultiTaskUNet3D
from train_3d_multi_augument import MultiTaskUNet3D, pair_ts_paths


# ============================
# fixed crop (train と一致)
# ============================
CROP_X = (50, 200)
CROP_Y = (45, 210)


def crop_xy(vol: np.ndarray) -> np.ndarray:
    x0, x1 = CROP_X
    y0, y1 = CROP_Y
    return vol[x0:x1, y0:y1, :]


def case_id_from_label_path(lab_path: str) -> str:
    base = os.path.basename(lab_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


# ============================
# shape alignment (center crop)
# ============================
def center_crop_3d_to_match(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    assert a.ndim == 3 and b.ndim == 3
    da, ha, wa = a.shape
    db, hb, wb = b.shape
    dt, ht, wt = min(da, db), min(ha, hb), min(wa, wb)

    def crop(v, dt, ht, wt):
        d, h, w = v.shape
        ds = (d - dt) // 2
        hs = (h - ht) // 2
        ws = (w - wt) // 2
        return v[ds : ds + dt, hs : hs + ht, ws : ws + wt]

    return crop(a, dt, ht, wt), crop(b, dt, ht, wt)


def dice_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred, gt = center_crop_3d_to_match(pred, gt)
    pred = pred.astype(np.float32).ravel()
    gt = gt.astype(np.float32).ravel()
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum() + eps)
    return float(2.0 * inter / denom)


# ============================
# surface metrics
# ============================
def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    er = ndi.binary_erosion(mask, iterations=1)
    return mask ^ er


def hd95_mm(
    pred: np.ndarray, gt: np.ndarray, spacing_mm: Tuple[float, float, float]
) -> float:
    pred, gt = center_crop_3d_to_match(pred, gt)
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

    d_p2g = dt_gt[pred_s]
    d_g2p = dt_pred[gt_s]
    all_d = np.concatenate([d_p2g, d_g2p]).astype(np.float32)
    if all_d.size == 0:
        return 0.0
    return float(np.percentile(all_d, 95))


def asd_mm(
    pred: np.ndarray, gt: np.ndarray, spacing_mm: Tuple[float, float, float]
) -> float:
    pred, gt = center_crop_3d_to_match(pred, gt)
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

    d_p2g = dt_gt[pred_s]
    d_g2p = dt_pred[gt_s]
    if d_p2g.size == 0 and d_g2p.size == 0:
        return 0.0
    return float((d_p2g.mean() + d_g2p.mean()) / 2.0)


def boundary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = center_crop_3d_to_match(pred, gt)
    pred_s = _surface(pred.astype(bool))
    gt_s = _surface(gt.astype(bool))
    inter = int(np.logical_and(pred_s, gt_s).sum())
    union = int(np.logical_or(pred_s, gt_s).sum())
    if union == 0:
        return 1.0
    return float(inter / union)


# ============================
# CSV writer
# ============================
def save_csv(rows: List[Dict], csv_path: str):
    import csv

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    # ★要求通りの列（root/dura両方欲しいので列名を増やす）
    fieldnames = [
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
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ============================
# main test
# ============================
@torch.no_grad()
def run_test(
    dataset_root: str,
    checkpoint_path: str,
    out_dir: str,
    nerve_root_label: int = 1,
    dura_label: int = 2,
    threshold: float = 0.5,
    spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    save_pred: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ts_imgs, ts_labs = pair_ts_paths(dataset_root)
    print(f"#Test cases: {len(ts_imgs)}")

    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, "pred_nii")
    if save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    model = MultiTaskUNet3D(in_channels=1, base_channels=16).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    rows: List[Dict] = []

    root_dices, root_hd95s, root_asds, root_bious = [], [], [], []
    dura_dices, dura_hd95s, dura_asds, dura_bious = [], [], [], []

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

        # train と一致：crop → normalize
        img_c = crop_xy(img)
        lab_c = crop_xy(lab)

        vmin, vmax = float(img_c.min()), float(img_c.max())
        if vmax > vmin:
            img_c = (img_c - vmin) / (vmax - vmin)
        else:
            img_c = np.zeros_like(img_c, dtype=np.float32)

        gt_root = (lab_c == nerve_root_label).astype(np.uint8)
        gt_dura = (lab_c == dura_label).astype(np.uint8)

        # inference
        vol = torch.from_numpy(img_c[None, ...]).unsqueeze(0).to(device)  # (1,1,X,Y,Z)
        logits_root, logits_dura = model(vol)

        pr_root = (
            (torch.sigmoid(logits_root) > threshold)
            .float()
            .cpu()
            .numpy()[0, 0]
            .astype(np.uint8)
        )
        pr_dura = (
            (torch.sigmoid(logits_dura) > threshold)
            .float()
            .cpu()
            .numpy()[0, 0]
            .astype(np.uint8)
        )

        # metrics (center-crop align inside each metric)
        rd = dice_coeff(pr_root, gt_root)
        rh = hd95_mm(pr_root, gt_root, spacing_mm=spacing_mm)
        ra = asd_mm(pr_root, gt_root, spacing_mm=spacing_mm)
        rb = boundary_iou(pr_root, gt_root)

        dd = dice_coeff(pr_dura, gt_dura)
        dh = hd95_mm(pr_dura, gt_dura, spacing_mm=spacing_mm)
        da = asd_mm(pr_dura, gt_dura, spacing_mm=spacing_mm)
        db = boundary_iou(pr_dura, gt_dura)

        print(
            f"Case {cid} | "
            f"root Dice={rd:.4f}, HD95={rh:.3f}, ASD={ra:.3f}, bIoU={rb:.4f} | "
            f"dura Dice={dd:.4f}, HD95={dh:.3f}, ASD={da:.3f}, bIoU={db:.4f}"
        )

        rows.append(
            {
                "case_id": cid,
                "root_dice": rd,
                "root_hd95_mm": rh,
                "root_asd_mm": ra,
                "root_boundary_iou": rb,
                "dura_dice": dd,
                "dura_hd95_mm": dh,
                "dura_asd_mm": da,
                "dura_boundary_iou": db,
            }
        )

        root_dices.append(rd)
        root_hd95s.append(rh)
        root_asds.append(ra)
        root_bious.append(rb)
        dura_dices.append(dd)
        dura_hd95s.append(dh)
        dura_asds.append(da)
        dura_bious.append(db)

        # optional: save predictions (cropped space)
        if save_pred:
            # 保存は「cropped空間」で良い（trainと一致）。必要なら full へ戻す処理も書ける。
            pr_root_nii = nib.Nifti1Image(pr_root.astype(np.uint8), np.eye(4))
            pr_dura_nii = nib.Nifti1Image(pr_dura.astype(np.uint8), np.eye(4))
            nib.save(
                pr_root_nii, os.path.join(pred_dir, f"{cid}_root_pred_crop.nii.gz")
            )
            nib.save(
                pr_dura_nii, os.path.join(pred_dir, f"{cid}_dura_pred_crop.nii.gz")
            )

    # MEAN row
    def _mean(xs):
        return float(np.mean(xs)) if len(xs) else ""

    rows.append(
        {
            "case_id": "MEAN",
            "root_dice": _mean(root_dices),
            "root_hd95_mm": _mean(root_hd95s),
            "root_asd_mm": _mean(root_asds),
            "root_boundary_iou": _mean(root_bious),
            "dura_dice": _mean(dura_dices),
            "dura_hd95_mm": _mean(dura_hd95s),
            "dura_asd_mm": _mean(dura_asds),
            "dura_boundary_iou": _mean(dura_bious),
        }
    )

    csv_path = os.path.join(out_dir, "test_metrics.csv")
    save_csv(rows, csv_path)
    print(f"\nSaved CSV: {csv_path}")


def main():
    p = argparse.ArgumentParser(
        description="MultiTask 3D U-Net test (root+dura) -> CSV"
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./test_out_mt")
    p.add_argument("--nerve_root_label", type=int, default=1)
    p.add_argument("--dura_label", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--spacing_mm", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    p.add_argument("--save_pred", action="store_true")
    args = p.parse_args()

    run_test(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        nerve_root_label=args.nerve_root_label,
        dura_label=args.dura_label,
        threshold=args.threshold,
        spacing_mm=tuple(args.spacing_mm),
        save_pred=args.save_pred,
    )


if __name__ == "__main__":
    main()
