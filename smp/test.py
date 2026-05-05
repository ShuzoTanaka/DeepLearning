# -*- coding: utf-8 -*-
"""
Test script for 2D axial slice inference using a trained model.
Saves:
    - Per-case metrics (Dice / HD95 / ASD / Boundary IoU)
    - Per-class mean metrics (nerve / spinal)
    - Overall mean across all classes
    - Slice-level predicted PNGs
    - 3D NIfTI prediction per case
    - CSV of metrics (RESULT_DIR/metrics.csv)
"""

import os
import glob
import csv
import numpy as np
import nibabel as nib
import cv2
from datetime import datetime

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from scipy import ndimage as ndi


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"

DATA_ROOT = r"C:\Users\orilab\Desktop\masumoto\smp\Dataset001_lumber"
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

BEST_MODEL_PATH = r"C:\Users\orilab\Desktop\masumoto\smp\checkpoints\20251207_1717_unet_resnet34_nifti2d.pth"

RESULT_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\results1208"
os.makedirs(RESULT_DIR, exist_ok=True)

METRICS_CSV_PATH = os.path.join(RESULT_DIR, "metrics.csv")

IMAGE_SIZE = 256
spacing = (3.0, 1.25, 1.25)  # voxel spacing (z,y,x) → 必要なら変更


# =========================
# Utility
# =========================
def to_tensor(x):
    return x.transpose(2, 0, 1).astype("float32")


def strip_nii_ext(fname):
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return os.path.splitext(fname)[0]


def find_test_cases(images_dir, labels_dir):
    image_files = glob.glob(os.path.join(images_dir, "*.nii*"))
    cases = {}

    for img_path in sorted(image_files):
        base = os.path.basename(img_path)
        stem = strip_nii_ext(base)

        if not stem.endswith("_0000"):
            continue

        case_id = stem[:-5]

        label1 = os.path.join(labels_dir, case_id + ".nii.gz")
        label2 = os.path.join(labels_dir, case_id + ".nii")

        if os.path.exists(label1):
            cases[case_id] = (img_path, label1)
        elif os.path.exists(label2):
            cases[case_id] = (img_path, label2)

    return cases


# =========================
# Metrics
# =========================
def dice_3d(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() and not pred.any():
        return None
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 0
    return 2 * inter / denom


def surface_distances(gt, pred, spacing):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() or not pred.any():
        return None

    struct = ndi.generate_binary_structure(3, 1)
    gt_b = np.logical_and(gt, ~ndi.binary_erosion(gt, struct))
    pr_b = np.logical_and(pred, ~ndi.binary_erosion(pred, struct))

    dt_gt = ndi.distance_transform_edt(~gt_b, sampling=spacing)
    dt_pr = ndi.distance_transform_edt(~pr_b, sampling=spacing)

    d1 = dt_gt[pr_b]
    d2 = dt_pr[gt_b]
    return np.concatenate([d1, d2])


def hd95_asd(gt, pred, spacing):
    sd = surface_distances(gt, pred, spacing)
    if sd is None or len(sd) == 0:
        return None, None
    return np.percentile(sd, 95), np.mean(sd)


def boundary_iou(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() and not pred.any():
        return None

    struct = ndi.generate_binary_structure(3, 1)
    gt_b = np.logical_and(gt, ~ndi.binary_erosion(gt, struct))
    pr_b = np.logical_and(pred, ~ndi.binary_erosion(pred, struct))

    gt_d = ndi.binary_dilation(gt_b, struct)
    pr_d = ndi.binary_dilation(pr_b, struct)

    inter = np.logical_and(gt_d, pr_d).sum()
    union = np.logical_or(gt_d, pr_d).sum()
    if union == 0:
        return 0
    return inter / union


# =========================
# Main
# =========================
def main():

    print("Loading model:", BEST_MODEL_PATH)
    model = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model = model.to(DEVICE)
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    cases = find_test_cases(IMAGES_TS_DIR, LABELS_TS_DIR)
    case_ids = sorted(cases.keys())

    print("Test cases:", case_ids)

    # per-class accumulation
    stats_all = {
        "nerve": {"dice": [], "hd95": [], "asd": [], "biou": []},
        "spinal": {"dice": [], "hd95": [], "asd": [], "biou": []},
    }

    class_map = {"nerve": 1, "spinal": 2}

    # ==============================
    # CSV 初期化
    # ==============================
    with open(METRICS_CSV_PATH, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["case_id", "class", "dice", "hd95", "asd", "boundary_iou"])

        # ======================================
        # Case loop
        # ======================================
        for cid in case_ids:
            print("----------------------------------------------------------")
            print(f"[{cid}] evaluating...")

            img_path, lbl_path = cases[cid]

            img_vol = nib.load(img_path).get_fdata()
            lbl_vol = nib.load(lbl_path).get_fdata().astype(np.int16)

            if img_vol.ndim == 4:
                img_vol = img_vol[..., 0]
            if lbl_vol.ndim == 4:
                lbl_vol = lbl_vol[..., 0]

            H, W, Z = img_vol.shape

            pred_vol = np.zeros((H, W, Z), dtype=np.int16)

            save_dir = os.path.join(RESULT_DIR, cid)
            os.makedirs(save_dir, exist_ok=True)

            # ------------- inference (slice-wise) -------------
            with torch.no_grad():
                for z in range(Z):
                    sl = img_vol[:, :, z].astype(np.float32)
                    mn, mx = sl.min(), sl.max()
                    if mx > mn:
                        sl_n = (sl - mn) / (mx - mn)
                    else:
                        sl_n = np.zeros_like(sl)

                    sl_u8 = (sl_n * 255).astype(np.uint8)
                    img3 = np.stack([sl_u8] * 3, axis=-1)

                    prep = preprocessing_fn(img3).astype("float32")
                    x = torch.from_numpy(to_tensor(prep)).unsqueeze(0).to(DEVICE)

                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    pr = probs.argmax(1).cpu().numpy()[0]

                    pred_vol[:, :, z] = pr.astype(np.int16)

                    # ---------- save PNG ----------
                    color_pred = np.zeros((H, W, 3), np.uint8)
                    color_pred[pred_vol[:, :, z] == 1] = (0, 255, 0)  # nerve: green
                    color_pred[pred_vol[:, :, z] == 2] = (0, 0, 255)  # spinal: red

                    overlay = cv2.addWeighted(
                        cv2.cvtColor(sl_u8, cv2.COLOR_GRAY2BGR), 0.6, color_pred, 0.4, 0
                    )

                    cv2.imwrite(
                        os.path.join(save_dir, f"slice_{z:03d}_pred.png"), color_pred
                    )
                    cv2.imwrite(
                        os.path.join(save_dir, f"slice_{z:03d}_overlay.png"), overlay
                    )

            # ------------- save NIfTI -------------
            out_nii_path = os.path.join(RESULT_DIR, f"{cid}_pred.nii.gz")
            nib.save(nib.Nifti1Image(pred_vol, affine=np.eye(4)), out_nii_path)
            print(f"Saved 3D prediction: {out_nii_path}")

            # ------------- metrics per class (write to CSV) -------------
            for cls, idx in class_map.items():
                gt = lbl_vol == idx
                pr = pred_vol == idx

                d = dice_3d(gt, pr)
                h, a = hd95_asd(gt, pr, spacing)
                b = boundary_iou(gt, pr)

                print(f"  [{cls}] Dice={d}, HD95={h}, ASD={a}, Boundary IoU={b}")

                # CSV 出力用に None → "" or "nan" とかに揃える
                def safe(v):
                    return "" if v is None else float(v)

                writer.writerow([cid, cls, safe(d), safe(h), safe(a), safe(b)])

                if d is not None:
                    stats_all[cls]["dice"].append(d)
                if h is not None:
                    stats_all[cls]["hd95"].append(h)
                if a is not None:
                    stats_all[cls]["asd"].append(a)
                if b is not None:
                    stats_all[cls]["biou"].append(b)

        # ======================================
        # 全体平均もCSVに書き込み
        # ======================================
        print("\n================== OVERALL METRICS ==================")

        def mean_or_nan(arr):
            return float(np.mean(arr)) if len(arr) else float("nan")

        # クラス別 mean
        for cls in ["nerve", "spinal"]:
            m_d = mean_or_nan(stats_all[cls]["dice"])
            m_h = mean_or_nan(stats_all[cls]["hd95"])
            m_a = mean_or_nan(stats_all[cls]["asd"])
            m_b = mean_or_nan(stats_all[cls]["biou"])

            print(
                f"[{cls}]  mean Dice={m_d:.4f}, "
                f"HD95={m_h:.4f}, ASD={m_a:.4f}, Boundary IoU={m_b:.4f}"
            )

            writer.writerow(["MEAN", cls, m_d, m_h, m_a, m_b])

        # nerve + spinal の overall
        all_dice = stats_all["nerve"]["dice"] + stats_all["spinal"]["dice"]
        all_hd95 = stats_all["nerve"]["hd95"] + stats_all["spinal"]["hd95"]
        all_asd = stats_all["nerve"]["asd"] + stats_all["spinal"]["asd"]
        all_biou = stats_all["nerve"]["biou"] + stats_all["spinal"]["biou"]

        ov_d = mean_or_nan(all_dice)
        ov_h = mean_or_nan(all_hd95)
        ov_a = mean_or_nan(all_asd)
        ov_b = mean_or_nan(all_biou)

        print(
            f"[overall] mean Dice={ov_d:.4f}, "
            f"HD95={ov_h:.4f}, ASD={ov_a:.4f}, Boundary IoU={ov_b:.4f}"
        )

        writer.writerow(["MEAN", "overall", ov_d, ov_h, ov_a, ov_b])

    print(f"\nMetrics CSV saved to: {METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
