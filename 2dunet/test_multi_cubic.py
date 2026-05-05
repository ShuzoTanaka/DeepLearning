# test_3d_multi_isotropic.py
import os
import argparse
import csv
from typing import Tuple, Dict, List

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch

# metrics用
try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
    import scipy.ndimage as ndi
except ImportError as e:
    raise ImportError("この test には scipy が必要です: pip install scipy") from e


# =========================================================
# モデルは train_3d_multi_isotropic.py から import 推奨
# =========================================================
from multi_cubic_train import MultiTaskUNet3D  # ← ファイル名が違うなら修正


# =========================================================
# Utility: file pairing (imagesTs/labelsTs)
# =========================================================
def pair_ts_paths(dataset_root: str) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(dataset_root, "imagesTs")
    lab_dir = os.path.join(dataset_root, "labelsTs")
    if not os.path.exists(lab_dir):
        raise RuntimeError("labelsTs が存在しません（テスト用GTが必要）")

    label_files = [
        f for f in os.listdir(lab_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    label_files.sort()

    image_paths, label_paths = [], []
    for lf in label_files:
        case_id = lf.replace(".nii.gz", "").replace(".nii", "")
        img_name = f"{case_id}_0000.nii.gz"
        img_path = os.path.join(img_dir, img_name)
        lab_path = os.path.join(lab_dir, lf)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Test image not found for label {lf}: expected {img_path}"
            )
        image_paths.append(img_path)
        label_paths.append(lab_path)

    return image_paths, label_paths


def case_id_from_label_path(lab_path: str) -> str:
    base = os.path.basename(lab_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


# =========================================================
# Resampling helpers
# =========================================================
def resample_img_lab_to_spacing(
    img: np.ndarray,
    lab: np.ndarray,
    orig_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """img: float32 (X,Y,Z), lab: int16 (X,Y,Z)"""
    sx, sy, sz = orig_spacing
    tx, ty, tz = target_spacing
    zoom = (sx / tx, sy / ty, sz / tz)

    img_r = ndi.zoom(img, zoom=zoom, order=3).astype(np.float32)
    lab_r = ndi.zoom(lab, zoom=zoom, order=0).astype(np.int16)
    return img_r, lab_r


def resample_mask_to_spacing(
    mask: np.ndarray,
    from_spacing: Tuple[float, float, float],
    to_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """mask: uint8(0/1) (X,Y,Z), 最近傍でspacing変換"""
    fx, fy, fz = from_spacing
    tx, ty, tz = to_spacing
    zoom = (fx / tx, fy / ty, fz / tz)
    out = ndi.zoom(mask.astype(np.uint8), zoom=zoom, order=0).astype(np.uint8)
    return out


def center_crop_or_pad_to_shape(
    vol: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """ndi.zoom の丸め差を吸収：中心crop/padで target_shape に合わせる"""
    x, y, z = vol.shape
    tx, ty, tz = target_shape

    out = np.zeros(target_shape, dtype=vol.dtype)

    # src範囲（中心）
    xs = max((x - tx) // 2, 0)
    ys = max((y - ty) // 2, 0)
    zs = max((z - tz) // 2, 0)

    # dst範囲（中心）
    xd = max((tx - x) // 2, 0)
    yd = max((ty - y) // 2, 0)
    zd = max((tz - z) // 2, 0)

    # コピーサイズ
    cx = min(x, tx)
    cy = min(y, ty)
    cz = min(z, tz)

    out[xd : xd + cx, yd : yd + cy, zd : zd + cz] = vol[
        xs : xs + cx, ys : ys + cy, zs : zs + cz
    ]
    return out


def embed_crop_back_fixed(
    crop_vol: np.ndarray,
    full_shape: Tuple[int, int, int],
    crop_x: Tuple[int, int],
    crop_y: Tuple[int, int],
) -> np.ndarray:
    """固定crop(50:200,45:210)で元fullへ埋め戻す"""
    out = np.zeros(full_shape, dtype=crop_vol.dtype)
    x0, x1 = crop_x
    y0, y1 = crop_y
    # zは全範囲を想定
    out[x0:x1, y0:y1, : crop_vol.shape[2]] = crop_vol
    return out


# =========================================================
# Metrics
# =========================================================
def boundary_mask(binmask: np.ndarray) -> np.ndarray:
    binmask = (binmask > 0).astype(bool)
    if binmask.sum() == 0:
        return np.zeros_like(binmask, dtype=bool)
    er = binary_erosion(binmask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    return binmask & (~er)


def dice_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum()) + eps
    return 2.0 * inter / denom


def surface_distances_mm(
    a: np.ndarray, b: np.ndarray, spacing: Tuple[float, float, float]
) -> np.ndarray:
    a_bd = boundary_mask(a)
    b = (b > 0).astype(bool)

    if a_bd.sum() == 0:
        return np.array([], dtype=np.float32)
    if b.sum() == 0:
        return np.array([np.inf], dtype=np.float32)

    dt = distance_transform_edt(~b, sampling=spacing)  # mm
    return dt[a_bd].astype(np.float32)


def hd95_asd_mm(
    pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[float, float]:
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf"), float("inf")

    d1 = surface_distances_mm(pred, gt, spacing)
    d2 = surface_distances_mm(gt, pred, spacing)
    d = np.concatenate([d1, d2], axis=0)

    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("inf"), float("inf")

    return float(np.percentile(d, 95)), float(d.mean())


def boundary_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pb = boundary_mask(pred > 0)
    gb = boundary_mask(gt > 0)
    inter = float((pb & gb).sum())
    union = float((pb | gb).sum()) + eps
    return inter / union


# =========================================================
# Core
# =========================================================
def load_case_and_preprocess(
    img_path: str,
    lab_path: str,
    crop_x: Tuple[int, int],
    crop_y: Tuple[int, int],
    target_spacing: Tuple[float, float, float],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    nib.Nifti1Image,
    Tuple[float, float, float],
    Tuple[int, int, int],
]:
    """
    return:
      img_iso_norm: (Xi,Yi,Zi) float32 0-1
      lab_iso:      (Xi,Yi,Zi) int16
      img_nii:      nib object (元affine/header用)
      orig_spacing: (sx,sy,sz)
      crop_shape_orig: (Xc,Yc,Zc)  # 元spacingでのcropサイズ
    """
    img_nii = nib.load(img_path)
    lab_nii = nib.load(lab_path)

    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.int16)

    if img.shape != lab.shape:
        raise ValueError(f"Shape mismatch: img {img.shape} vs lab {lab.shape}")

    zooms = img_nii.header.get_zooms()[:3]
    orig_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    # fixed crop (元spacingのまま)
    x0, x1 = crop_x
    y0, y1 = crop_y
    img_c = img[x0:x1, y0:y1, :]
    lab_c = lab[x0:x1, y0:y1, :]
    crop_shape_orig = img_c.shape  # (Xc,Yc,Zc)

    # resample to target spacing
    img_iso, lab_iso = resample_img_lab_to_spacing(
        img_c, lab_c, orig_spacing, target_spacing
    )

    # resample後の丸め差でshapeがズレることがあるので、必ず揃える
    lab_iso = center_crop_or_pad_to_shape(lab_iso, img_iso.shape)
    # あるいは逆でも良いが、推論は img で走るので lab を img に合わせるのが自然

    # normalize (iso)
    vmin, vmax = float(img_iso.min()), float(img_iso.max())
    if vmax > vmin:
        img_iso = (img_iso - vmin) / (vmax - vmin)
    else:
        img_iso = np.zeros_like(img_iso, dtype=np.float32)

    return (
        img_iso.astype(np.float32),
        lab_iso.astype(np.int16),
        img_nii,
        orig_spacing,
        crop_shape_orig,
    )


def run_test(
    dataset_root: str,
    checkpoint: str,
    out_dir: str,
    root_label: int,
    dura_label: int,
    thr_root: float,
    thr_dura: float,
    target_spacing: Tuple[float, float, float],
    crop_x: Tuple[int, int],
    crop_y: Tuple[int, int],
    save_nifti: bool,
    report_original_space: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ts_imgs, ts_labs = pair_ts_paths(dataset_root)
    print(f"#Test cases: {len(ts_imgs)}")
    os.makedirs(out_dir, exist_ok=True)

    # load model
    model = MultiTaskUNet3D(in_channels=1, base_channels=16).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    model.eval()

    rows = []
    rows_orig = []

    for img_path, lab_path in tqdm(
        list(zip(ts_imgs, ts_labs)), desc="Test", leave=False
    ):
        cid = case_id_from_label_path(lab_path)

        img_iso, lab_iso, img_nii, orig_spacing, crop_shape_orig = (
            load_case_and_preprocess(img_path, lab_path, crop_x, crop_y, target_spacing)
        )

        # GT masks in iso space（いったん img_iso と同shape のはず）
        gt_root_iso = (lab_iso == root_label).astype(np.uint8)
        gt_dura_iso = (lab_iso == dura_label).astype(np.uint8)

        # inference
        vol = torch.from_numpy(img_iso[None, None, ...]).to(device)  # (1,1,X,Y,Z)
        with torch.no_grad():
            logits_root, logits_dura = model(vol)

        pr_root_iso = (
            torch.sigmoid(logits_root)[0, 0].cpu().numpy() > thr_root
        ).astype(np.uint8)
        pr_dura_iso = (
            torch.sigmoid(logits_dura)[0, 0].cpu().numpy() > thr_dura
        ).astype(np.uint8)

        # ==========================================================
        # ★最重要：pred と GT を必ず同じ shape に揃える（pred基準）
        #    UNetのcrop等で pred が小さくなることがあるため
        # ==========================================================
        gt_root_iso = center_crop_or_pad_to_shape(gt_root_iso, pr_root_iso.shape)
        gt_dura_iso = center_crop_or_pad_to_shape(gt_dura_iso, pr_dura_iso.shape)

        # ISO metrics (mm with target_spacing)
        root_dice = dice_coeff(pr_root_iso, gt_root_iso)
        root_hd95, root_asd = hd95_asd_mm(pr_root_iso, gt_root_iso, target_spacing)
        root_biou = boundary_iou(pr_root_iso, gt_root_iso)

        dura_dice = dice_coeff(pr_dura_iso, gt_dura_iso)
        dura_hd95, dura_asd = hd95_asd_mm(pr_dura_iso, gt_dura_iso, target_spacing)
        dura_biou = boundary_iou(pr_dura_iso, gt_dura_iso)

        rows.append(
            dict(
                case_id=cid,
                root_dice=float(root_dice),
                root_hd95_mm=float(root_hd95),
                root_asd_mm=float(root_asd),
                root_boundary_iou=float(root_biou),
                dura_dice=float(dura_dice),
                dura_hd95_mm=float(dura_hd95),
                dura_asd_mm=float(dura_asd),
                dura_boundary_iou=float(dura_biou),
            )
        )

        print(
            f"\nCase {cid} | "
            f"root Dice={root_dice:.4f}, HD95={root_hd95:.3f}mm, ASD={root_asd:.3f}mm, bIoU={root_biou:.4f} | "
            f"dura Dice={dura_dice:.4f}, HD95={dura_hd95:.3f}mm, ASD={dura_asd:.3f}mm, bIoU={dura_biou:.4f}"
        )

        # ----------------------------------------------------------
        # save NIfTI (original geometry)
        # ----------------------------------------------------------
        if save_nifti:
            # iso pred -> back to orig-crop spacing（最近傍）
            pr_root_back = resample_mask_to_spacing(
                pr_root_iso, from_spacing=target_spacing, to_spacing=orig_spacing
            )
            pr_dura_back = resample_mask_to_spacing(
                pr_dura_iso, from_spacing=target_spacing, to_spacing=orig_spacing
            )

            # 丸め差を crop_shape_orig に吸収
            pr_root_back = center_crop_or_pad_to_shape(pr_root_back, crop_shape_orig)
            pr_dura_back = center_crop_or_pad_to_shape(pr_dura_back, crop_shape_orig)

            # embed to full
            full_shape = img_nii.shape
            root_full = embed_crop_back_fixed(pr_root_back, full_shape, crop_x, crop_y)
            dura_full = embed_crop_back_fixed(pr_dura_back, full_shape, crop_x, crop_y)

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

        # ----------------------------------------------------------
        # (optional) Original-space metrics
        # ----------------------------------------------------------
        if report_original_space:
            # GT crop in original spacing
            lab_full = nib.load(lab_path).get_fdata().astype(np.int16)
            gt_root_orig = (
                lab_full[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1], :] == root_label
            ).astype(np.uint8)
            gt_dura_orig = (
                lab_full[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1], :] == dura_label
            ).astype(np.uint8)

            # pred back to original spacing (crop space)
            pr_root_back = resample_mask_to_spacing(
                pr_root_iso, from_spacing=target_spacing, to_spacing=orig_spacing
            )
            pr_dura_back = resample_mask_to_spacing(
                pr_dura_iso, from_spacing=target_spacing, to_spacing=orig_spacing
            )

            # 丸め差を GT のshapeに吸収（GT基準）
            pr_root_back = center_crop_or_pad_to_shape(pr_root_back, gt_root_orig.shape)
            pr_dura_back = center_crop_or_pad_to_shape(pr_dura_back, gt_dura_orig.shape)

            root_dice_o = dice_coeff(pr_root_back, gt_root_orig)
            root_hd95_o, root_asd_o = hd95_asd_mm(
                pr_root_back, gt_root_orig, orig_spacing
            )
            root_biou_o = boundary_iou(pr_root_back, gt_root_orig)

            dura_dice_o = dice_coeff(pr_dura_back, gt_dura_orig)
            dura_hd95_o, dura_asd_o = hd95_asd_mm(
                pr_dura_back, gt_dura_orig, orig_spacing
            )
            dura_biou_o = boundary_iou(pr_dura_back, gt_dura_orig)

            rows_orig.append(
                dict(
                    case_id=cid,
                    root_dice=float(root_dice_o),
                    root_hd95_mm=float(root_hd95_o),
                    root_asd_mm=float(root_asd_o),
                    root_boundary_iou=float(root_biou_o),
                    dura_dice=float(dura_dice_o),
                    dura_hd95_mm=float(dura_hd95_o),
                    dura_asd_mm=float(dura_asd_o),
                    dura_boundary_iou=float(dura_biou_o),
                )
            )

    # ======================
    # Save CSV (ISO)
    # ======================
    csv_path = os.path.join(out_dir, "metrics_test_iso.csv")
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

        def mean_finite(vals):
            v = np.asarray(vals, dtype=np.float32)
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size > 0 else float("nan")

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

    print(f"\nSaved ISO metrics CSV: {csv_path}")
    print(f"ISO spacing used for HD95/ASD (mm): {target_spacing}")

    # ======================
    # Save CSV (Original) optional
    # ======================
    if report_original_space:
        csv_path_o = os.path.join(out_dir, "metrics_test_original.csv")
        with open(csv_path_o, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows_orig:
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

            def mean_finite(vals):
                v = np.asarray(vals, dtype=np.float32)
                v = v[np.isfinite(v)]
                return float(v.mean()) if v.size > 0 else float("nan")

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

        print(f"Saved ORIGINAL-space metrics CSV: {csv_path_o}")
        print("ORIGINAL spacing used for HD95/ASD (mm): per-case NIfTI header zooms")


def main():
    p = argparse.ArgumentParser(
        "MultiTask 3D U-Net test (isotropic pipeline) with metrics + CSV"
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./pred3d_mt_iso")

    p.add_argument("--root_label", type=int, default=1)
    p.add_argument("--dura_label", type=int, default=2)
    p.add_argument("--thr_root", type=float, default=0.5)
    p.add_argument("--thr_dura", type=float, default=0.5)

    p.add_argument("--target_spacing_mm", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p.add_argument("--crop_x", type=int, nargs=2, default=[50, 200])
    p.add_argument("--crop_y", type=int, nargs=2, default=[45, 210])

    p.add_argument("--no_save_nifti", action="store_true", help="予測NIfTIを保存しない")
    p.add_argument(
        "--report_original_space",
        action="store_true",
        help="予測を元spacingに戻して、orig spacing (例1.25,1.25,3.0) でHD95/ASDも計算してCSV保存",
    )

    args = p.parse_args()

    run_test(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        root_label=args.root_label,
        dura_label=args.dura_label,
        thr_root=args.thr_root,
        thr_dura=args.thr_dura,
        target_spacing=(
            float(args.target_spacing_mm[0]),
            float(args.target_spacing_mm[1]),
            float(args.target_spacing_mm[2]),
        ),
        crop_x=(int(args.crop_x[0]), int(args.crop_x[1])),
        crop_y=(int(args.crop_y[0]), int(args.crop_y[1])),
        save_nifti=not args.no_save_nifti,
        report_original_space=args.report_original_space,
    )


if __name__ == "__main__":
    main()
