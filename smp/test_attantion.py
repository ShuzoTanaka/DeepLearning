# -*- coding: utf-8 -*-
"""
Test script for 2D axial inference with Attention U-Net.

- 読み込み:
    Dataset001_lumber/imagesTs, labelsTs
- クラス:
    0: background
    1: nerve
    2: spinal
- 出力:
    - 各症例ごとの Dice / HD95 / ASD / Boundary IoU（nerve, spinal）
    - nerve / spinal / overall の平均
    - caseごとの 3D prediction NIfTI
    - 各スライスの pred PNG + overlay PNG
    - metrics.csv（case_id, class, dice, hd95, asd, boundary_iou）
"""

import os
import glob
import csv
import numpy as np
import nibabel as nib
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage as ndi


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = r"C:\Users\orilab\Desktop\masumoto\smp\Dataset001_lumber"
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

CHECKPOINT_PATH = (
    r"C:\Users\orilab\Desktop\masumoto\smp\checkpoints_attention\YOUR_ATT_MODEL.pth"
)

RESULT_DIR = r"C:\Users\orilab\Desktop\masumoto\smp\results_attention"
os.makedirs(RESULT_DIR, exist_ok=True)

METRICS_CSV_PATH = os.path.join(RESULT_DIR, "metrics.csv")

spacing = (1.0, 1.0, 1.0)  # (z, y, x) voxel size → DTI に合わせたいなら変更


# =========================
# Utility
# =========================
def strip_nii_ext(fname: str) -> str:
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return os.path.splitext(fname)[0]


def find_test_cases(images_dir, labels_dir):
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    cases = {}

    for img_path in image_files:
        base = os.path.basename(img_path)
        stem = strip_nii_ext(base)

        if not stem.endswith("_0000"):
            continue

        case_id = stem[:-5]

        lab1 = os.path.join(labels_dir, case_id + ".nii.gz")
        lab2 = os.path.join(labels_dir, case_id + ".nii")

        if os.path.exists(lab1):
            cases[case_id] = (img_path, lab1)
        elif os.path.exists(lab2):
            cases[case_id] = (img_path, lab2)

    return cases


def to_tensor(x):
    return x.transpose(2, 0, 1).astype("float32")


# =========================
# Attention U-Net 2D（train.py と同じ定義）
# =========================
class AttentionBlock2D(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(512, 1024)

        self.att4 = AttentionBlock2D(512, 512, 256)
        self.att3 = AttentionBlock2D(256, 256, 128)
        self.att2 = AttentionBlock2D(128, 128, 64)
        self.att1 = AttentionBlock2D(64, 64, 32)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        center = self.center(p4)

        d4 = self.up4(center)
        e4_att = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        logits = self.seg_head(d1)
        return logits


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
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", DEVICE)

    # --- model load (state_dict) ---
    model = AttentionUNet2D(in_channels=3, num_classes=3).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("Loaded checkpoint:", CHECKPOINT_PATH)

    # --- find cases ---
    cases = find_test_cases(IMAGES_TS_DIR, LABELS_TS_DIR)
    case_ids = sorted(cases.keys())
    print("Test cases:", case_ids)

    class_map = {"nerve": 1, "spinal": 2}

    # 全症例の集計用
    stats_all = {
        "nerve": {"dice": [], "hd95": [], "asd": [], "biou": []},
        "spinal": {"dice": [], "hd95": [], "asd": [], "biou": []},
    }

    with open(METRICS_CSV_PATH, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["case_id", "class", "dice", "hd95", "asd", "boundary_iou"])

        for cid in case_ids:
            print("----------------------------------------------------------")
            print(f"[{cid}] evaluating...")

            img_path, lab_path = cases[cid]

            img_vol = nib.load(img_path).get_fdata()
            lab_vol = nib.load(lab_path).get_fdata().astype(np.int16)

            if img_vol.ndim == 4:
                img_vol = img_vol[..., 0]
            if lab_vol.ndim == 4:
                lab_vol = lab_vol[..., 0]

            H, W, Z = img_vol.shape
            pred_vol = np.zeros((H, W, Z), dtype=np.int16)

            save_dir = os.path.join(RESULT_DIR, cid)
            os.makedirs(save_dir, exist_ok=True)

            # ---- slice-wise inference + PNG 保存 ----
            with torch.no_grad():
                for z in range(Z):
                    sl = img_vol[:, :, z].astype(np.float32)
                    mn, mx = sl.min(), sl.max()
                    if mx > mn:
                        sl_n = (sl - mn) / (mx - mn)
                    else:
                        sl_n = np.zeros_like(sl)

                    sl_u8 = (sl_n * 255.0).astype(np.uint8)
                    img3 = np.stack([sl_u8] * 3, axis=-1)

                    x = torch.from_numpy(to_tensor(img3)).unsqueeze(0).to(DEVICE)
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    pr = probs.argmax(1).cpu().numpy()[0]

                    pred_vol[:, :, z] = pr.astype(np.int16)

                    # PNG (pred + overlay)
                    color_pred = np.zeros((H, W, 3), np.uint8)
                    color_pred[pred_vol[:, :, z] == 1] = (0, 255, 0)  # nerve
                    color_pred[pred_vol[:, :, z] == 2] = (0, 0, 255)  # spinal

                    overlay = cv2.addWeighted(
                        cv2.cvtColor(sl_u8, cv2.COLOR_GRAY2BGR), 0.6, color_pred, 0.4, 0
                    )

                    cv2.imwrite(
                        os.path.join(save_dir, f"slice_{z:03d}_pred.png"), color_pred
                    )
                    cv2.imwrite(
                        os.path.join(save_dir, f"slice_{z:03d}_overlay.png"), overlay
                    )

            # ---- save 3D NIfTI prediction ----
            out_nii = os.path.join(RESULT_DIR, f"{cid}_pred.nii.gz")
            nib.save(nib.Nifti1Image(pred_vol, affine=np.eye(4)), out_nii)
            print("Saved NIfTI:", out_nii)

            # ---- metrics per class ----
            for cls, idx in class_map.items():
                gt = lab_vol == idx
                pr = pred_vol == idx

                d = dice_3d(gt, pr)
                h, a = hd95_asd(gt, pr, spacing)
                b = boundary_iou(gt, pr)

                print(f"  [{cls}] Dice={d}, HD95={h}, ASD={a}, Boundary IoU={b}")

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

        # ---- 全体平均 ----
        def mean_or_nan(arr):
            return float(np.mean(arr)) if len(arr) else float("nan")

        print("\n================== OVERALL METRICS ==================")
        for cls in ["nerve", "spinal"]:
            m_d = mean_or_nan(stats_all[cls]["dice"])
            m_h = mean_or_nan(stats_all[cls]["hd95"])
            m_a = mean_or_nan(stats_all[cls]["asd"])
            m_b = mean_or_nan(stats_all[cls]["biou"])

            print(
                f"[{cls}] mean Dice={m_d:.4f}, HD95={m_h:.4f}, ASD={m_a:.4f}, Boundary IoU={m_b:.4f}"
            )
            writer.writerow(["MEAN", cls, m_d, m_h, m_a, m_b])

        all_dice = stats_all["nerve"]["dice"] + stats_all["spinal"]["dice"]
        all_hd95 = stats_all["nerve"]["hd95"] + stats_all["spinal"]["hd95"]
        all_asd = stats_all["nerve"]["asd"] + stats_all["spinal"]["asd"]
        all_biou = stats_all["nerve"]["biou"] + stats_all["spinal"]["biou"]

        ov_d = mean_or_nan(all_dice)
        ov_h = mean_or_nan(all_hd95)
        ov_a = mean_or_nan(all_asd)
        ov_b = mean_or_nan(all_biou)

        print(
            f"[overall] mean Dice={ov_d:.4f}, HD95={ov_h:.4f}, ASD={ov_a:.4f}, Boundary IoU={ov_b:.4f}"
        )
        writer.writerow(["MEAN", "overall", ov_d, ov_h, ov_a, ov_b])

    print(f"\nMetrics CSV saved to: {METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
