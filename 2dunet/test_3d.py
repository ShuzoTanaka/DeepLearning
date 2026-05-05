import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm

import csv


import torch

# train_3d.py から必要なものをインポート
from train_3d import UNet3D, pair_ts_paths, dice_coeff_numpy

# ★ 追加：HD95 / ASD / Boundary IoU 用（SciPyが必要）
try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception as e:
    raise ImportError(
        "HD95/ASD/BoundaryIoU の計算には SciPy が必要です。"
        " `pip install scipy` を実行してから再実行してください。"
    ) from e


def load_volume_and_mask(
    img_path: str,
    lab_path: str,
    nerve_root_label: int,
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """
    1症例分の画像と神経根GTマスクを読み込み＆前処理
    返り値:
      img_norm: 正規化済み画像 (X,Y,Z) float32
      nerve_mask: 神経根だけ1のGT (X,Y,Z) float32
      img_nii: nibabel NIfTIオブジェクト（affine & header用）
    """
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)

    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img_norm = (img - vmin) / (vmax - vmin)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)

    lab_nii = nib.load(lab_path)
    lab = lab_nii.get_fdata().astype(np.int16)
    if img.shape != lab.shape:
        raise ValueError(f"Shape mismatch: img {img.shape} vs lab {lab.shape}")

    nerve_mask = (lab == nerve_root_label).astype(np.uint8)  # 0/1

    return img_norm, nerve_mask, img_nii


def embed_center(
    small: np.ndarray,
    full_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    small: (D',H',W') を full_shape (X,Y,Z) の中心に0埋めで埋め込む
    """
    d_f, h_f, w_f = full_shape
    d_s, h_s, w_s = small.shape

    if d_s > d_f or h_s > h_f or w_s > w_f:
        raise ValueError(f"small volume {small.shape} is larger than full {full_shape}")

    out = np.zeros(full_shape, dtype=small.dtype)
    d_start = (d_f - d_s) // 2
    h_start = (h_f - h_s) // 2
    w_start = (w_f - w_s) // 2
    out[d_start : d_start + d_s, h_start : h_start + h_s, w_start : w_start + w_s] = (
        small
    )
    return out


def case_id_from_label_path(lab_path: str) -> str:
    base = os.path.basename(lab_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def center_crop_3d_to_match(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    a, b: (X,Y,Z)
    → minサイズに合わせて中心クロップして揃える
    """
    assert a.ndim == 3 and b.ndim == 3
    ax, ay, az = a.shape
    bx, by, bz = b.shape

    tx, ty, tz = min(ax, bx), min(ay, by), min(az, bz)

    def crop(v, tx, ty, tz):
        x, y, z = v.shape
        xs = (x - tx) // 2
        ys = (y - ty) // 2
        zs = (z - tz) // 2
        return v[xs : xs + tx, ys : ys + ty, zs : zs + tz]

    return crop(a, tx, ty, tz), crop(b, tx, ty, tz)


def boundary_mask(mask01: np.ndarray) -> np.ndarray:
    """
    1voxel厚の境界（mask XOR eroded(mask)）
    """
    mask = (mask01 > 0).astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    er = binary_erosion(mask, iterations=1)
    bd = mask ^ er
    return bd


def surface_distances_mm(
    pred01: np.ndarray, gt01: np.ndarray, spacing_xyz: Tuple[float, float, float]
) -> np.ndarray:
    """
    pred/gt の境界点から相手境界までの距離（mm）の両方向を返す。
    """
    pred_bd = boundary_mask(pred01)
    gt_bd = boundary_mask(gt01)

    # 片方が空なら距離が定義しにくいので呼び出し側で処理する想定
    # 距離変換：相手の境界がFalseのところが距離0になるように ~bd を渡す
    dt_to_gt = distance_transform_edt(~gt_bd, sampling=spacing_xyz)
    dt_to_pred = distance_transform_edt(~pred_bd, sampling=spacing_xyz)

    d_pred_to_gt = dt_to_gt[pred_bd]  # pred境界→gt境界
    d_gt_to_pred = dt_to_pred[gt_bd]  # gt境界→pred境界

    return np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)


def compute_metrics(
    pred01: np.ndarray,
    gt01: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    eps: float = 1e-6,
) -> Tuple[float, float, float, float]:
    """
    戻り値: (dice, hd95, asd, boundary_iou)
    pred01/gt01: 0/1（またはbool）3D
    """
    pred01, gt01 = center_crop_3d_to_match(pred01, gt01)

    pred_bin = (pred01 > 0).astype(np.uint8)
    gt_bin = (gt01 > 0).astype(np.uint8)

    # Dice（既存関数と同じ考え方でOK）
    dice = float(dice_coeff_numpy(pred_bin, gt_bin))

    pred_sum = int(pred_bin.sum())
    gt_sum = int(gt_bin.sum())

    # 空マスクの扱い
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 0.0, 0.0, 1.0
    if pred_sum == 0 and gt_sum > 0:
        return 0.0, float("inf"), float("inf"), 0.0
    if pred_sum > 0 and gt_sum == 0:
        return 0.0, float("inf"), float("inf"), 0.0

    # HD95 / ASD
    dists = surface_distances_mm(pred_bin, gt_bin, spacing_xyz)
    if dists.size == 0:
        # 境界が作れないケース（極端に小さい等）
        hd95 = float("inf")
        asd = float("inf")
    else:
        hd95 = float(np.percentile(dists, 95))
        asd = float(np.mean(dists))

    # Boundary IoU
    pb = boundary_mask(pred_bin)
    gb = boundary_mask(gt_bin)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    boundary_iou = float(inter / (union + eps))

    return dice, hd95, asd, boundary_iou


def run_inference(
    dataset_root: str,
    checkpoint_path: str,
    out_dir: str,
    nerve_root_label: int = 1,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Ts データに対して推論・評価・予測NIfTI保存。
    返り値: {case_id: {"dice":..., "hd95":..., "asd":..., "biou":...}}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ts_imgs, ts_labs = pair_ts_paths(dataset_root)
    print(f"#Test cases: {len(ts_imgs)}")

    os.makedirs(out_dir, exist_ok=True)

    model = UNet3D(in_channels=1, base_channels=16).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    metrics_by_case: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for img_path, lab_path in tqdm(
            list(zip(ts_imgs, ts_labs)), desc="Inference", leave=False
        ):
            case_id = case_id_from_label_path(lab_path)
            print(f"\nCase: {case_id}")

            img_norm, nerve_mask_gt, img_nii = load_volume_and_mask(
                img_path, lab_path, nerve_root_label
            )

            # spacing（mm）。ヘッダのズームをそのまま使う（X,Y,Z順を想定）
            zooms = img_nii.header.get_zooms()
            spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

            vol = (
                torch.from_numpy(img_norm[None, ...]).unsqueeze(0).to(device)
            )  # (1,1,X,Y,Z)

            logits = model(vol)
            probs = torch.sigmoid(logits)
            preds_bin = (probs > threshold).float()
            pred_small = preds_bin.cpu().numpy()[0, 0].astype(np.uint8)  # (X',Y',Z')

            # メトリクス（center cropでshape差分吸収）
            dice, hd95, asd, biou = compute_metrics(
                pred_small, nerve_mask_gt, spacing_xyz
            )

            metrics_by_case[case_id] = {
                "dice": float(dice),
                "hd95": float(hd95),
                "asd": float(asd),
                "boundary_iou": float(biou),
            }

            print(f"  Dice        : {dice:.4f}")
            print(
                f"  HD95 [mm]    : {hd95:.4f}"
                if np.isfinite(hd95)
                else "  HD95 [mm]    : inf"
            )
            print(
                f"  ASD  [mm]    : {asd:.4f}"
                if np.isfinite(asd)
                else "  ASD  [mm]    : inf"
            )
            print(f"  Boundary IoU : {biou:.4f}")

            # 予測maskを元画像サイズに中心埋めして保存（必要なら）
            full_shape = img_norm.shape
            pred_full = embed_center(pred_small.astype(np.uint8), full_shape)
            pred_nii = nib.Nifti1Image(pred_full, img_nii.affine, img_nii.header)
            pred_nii.set_data_dtype(np.uint8)
            out_path = os.path.join(out_dir, f"{case_id}_pred.nii.gz")
            nib.save(pred_nii, out_path)
            print(f"  Saved prediction: {out_path}")

    # 全体平均（infを含むと平均がinfになるので、finiteのみで平均）
    dices = np.array([v["dice"] for v in metrics_by_case.values()], dtype=np.float32)
    hd95s = np.array([v["hd95"] for v in metrics_by_case.values()], dtype=np.float64)
    asds = np.array([v["asd"] for v in metrics_by_case.values()], dtype=np.float64)
    bious = np.array(
        [v["boundary_iou"] for v in metrics_by_case.values()], dtype=np.float32
    )

    def mean_finite(x):
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size > 0 else float("nan")

    print("\n=== Test Metrics (nerve root) ===")
    print(f"Mean Dice        : {float(dices.mean()):.4f}")
    print(f"Mean HD95 [mm]   : {mean_finite(hd95s):.4f}")
    print(f"Mean ASD  [mm]   : {mean_finite(asds):.4f}")
    print(f"Mean Boundary IoU: {float(bious.mean()):.4f}")

    # =========================
    # CSV 保存
    # =========================
    csv_path = os.path.join(out_dir, "metrics_test.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "dice", "hd95_mm", "asd_mm", "boundary_iou"])

        for cid, m in metrics_by_case.items():
            writer.writerow(
                [
                    cid,
                    f"{m['dice']:.6f}",
                    f"{m['hd95']:.6f}" if np.isfinite(m["hd95"]) else "inf",
                    f"{m['asd']:.6f}" if np.isfinite(m["asd"]) else "inf",
                    f"{m['boundary_iou']:.6f}",
                ]
            )

        # mean（finiteのみ）
        def mean_finite(arr):
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            return float(arr.mean()) if arr.size > 0 else float("nan")

        mean_dice = float(np.mean([v["dice"] for v in metrics_by_case.values()]))
        mean_hd95 = mean_finite([v["hd95"] for v in metrics_by_case.values()])
        mean_asd = mean_finite([v["asd"] for v in metrics_by_case.values()])
        mean_biou = float(
            np.mean([v["boundary_iou"] for v in metrics_by_case.values()])
        )

        writer.writerow(
            [
                "MEAN",
                f"{mean_dice:.6f}",
                f"{mean_hd95:.6f}",
                f"{mean_asd:.6f}",
                f"{mean_biou:.6f}",
            ]
        )

    print(f"\nSaved metrics CSV: {csv_path}")

    return metrics_by_case


def main():
    parser = argparse.ArgumentParser(
        description="3D U-Net nerve-root segmentation: test/inference script"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset フォルダ (直下に imagesTs, labelsTs がある場所)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./ckpt3d/best_3dunet_nerve.pth",
        help="学習済みモデルのパス (train_3d.py で保存した .pth)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./pred3d",
        help="予測NIfTIを保存するフォルダ",
    )
    parser.add_argument(
        "--nerve_root_label",
        type=int,
        default=1,
        help="マスク内で神経根を表すラベル値 (GT用)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="2値化しきい値 (確率 > threshold を1とする)",
    )
    args = parser.parse_args()

    run_inference(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        nerve_root_label=args.nerve_root_label,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
