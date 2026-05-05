# train_test_2d_unet_axial_multitask.py
# 2D U-Net（Axialスライス学習）で multi-task（root + dura）を学習し、
# 学習後に imagesTs/labelsTs を case-wise に推論→3Dに再構成して
# case_id,root_dice,root_hd95_mm,root_asd_mm,root_boundary_iou,dura_dice,dura_hd95_mm,dura_asd_mm,dura_boundary_iou
# を CSV に保存する “一発完結” スクリプトです。
#
# Dataset 形式（nnU-Net風）:
#   Dataset/
#     imagesTr/  case001_0000.nii.gz ...
#     labelsTr/  case001.nii.gz ...
#     imagesTs/  caseXXX_0000.nii.gz ...
#     labelsTs/  caseXXX.nii.gz ...
#
# 使い方例:
#   python train_test_2d_unet_axial_multitask.py --dataset_root Dataset --out_dir ./ckpt2d_mt
#
# 推奨:
#   --crop_x 50 200 --crop_y 45 210 （3D版と同じROIに合わせるなら）
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
from torch.utils.data import Dataset, DataLoader

# metrics/scipy
try:
    import scipy.ndimage as ndi
    from scipy.ndimage import distance_transform_edt, binary_erosion
except ImportError as e:
    raise ImportError("このスクリプトには scipy が必要です: pip install scipy") from e


# =========================================================
# I/O utilities (pairing)
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


def train_val_split_by_case(
    image_paths: List[str],
    label_paths: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    assert len(image_paths) == len(label_paths)
    n = len(image_paths)
    idx = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    def subset(lst, ii):
        return [lst[i] for i in ii]

    return (
        subset(image_paths, tr_idx),
        subset(label_paths, tr_idx),
        subset(image_paths, val_idx),
        subset(label_paths, val_idx),
    )


# =========================================================
# 2D U-Net (multi-task: root + dura)
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
        # encoder
        self.enc1 = DoubleConv2D(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv2D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv2D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv2D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv2D(base_channels * 8, base_channels * 16)

        # shared decoder (coarse)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = DoubleConv2D(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv2D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv2D(base_channels * 4, base_channels * 2)

        # fine branches
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
        # enc/ref: (B,C,H,W)
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

        # root
        u1r = self.up1_root(d2)
        e1cr, u1r = self._center_crop_2d(e1, u1r)
        d1r = self.dec1_root(torch.cat([u1r, e1cr], dim=1))
        out_root = self.out_root(d1r)

        # dura
        u1d = self.up1_dura(d2)
        e1cd, u1d = self._center_crop_2d(e1, u1d)
        d1d = self.dec1_dura(torch.cat([u1d, e1cd], dim=1))
        out_dura = self.out_dura(d1d)

        return out_root, out_dura


# =========================================================
# Dataset: 3D NIfTI -> axial slices (H,W) with case-wise split
# =========================================================
class AxialSliceDataset(Dataset):
    """
    3D volume を case ごとに読み、Axial（Z方向）スライス単位で学習する Dataset。
    返り値:
      img:  (1,H,W) float32
      mask: (2,H,W) float32  (ch0=root, ch1=dura)
      case_id: str
      z_index: int
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        root_label: int = 1,
        dura_label: int = 2,
        crop_x: Optional[Tuple[int, int]] = None,
        crop_y: Optional[Tuple[int, int]] = None,
        include_empty_slices: bool = True,
        cache_volumes: bool = True,
    ):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.root_label = root_label
        self.dura_label = dura_label
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.include_empty_slices = include_empty_slices
        self.cache_volumes = cache_volumes

        self.case_ids = [case_id_from_label_path(p) for p in self.label_paths]

        # cache: cid -> (img3d, lab3d, spacing(x,y,z))
        self._cache: Dict[
            str, Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]
        ] = {}

        # slice index list: [(case_i, z), ...]
        self.index: List[Tuple[int, int]] = []
        self._build_index()

    def _load_case(
        self, i: int
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        cid = self.case_ids[i]
        if self.cache_volumes and cid in self._cache:
            return self._cache[cid]

        img_nii = nib.load(self.image_paths[i])
        lab_nii = nib.load(self.label_paths[i])

        img = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)
        lab = lab_nii.get_fdata().astype(np.int16)

        if img.shape != lab.shape:
            raise ValueError(
                f"Shape mismatch for {cid}: img{img.shape} vs lab{lab.shape}"
            )

        # optional fixed crop in X,Y (match your 3D pipeline ROI)
        if self.crop_x is not None and self.crop_y is not None:
            x0, x1 = self.crop_x
            y0, y1 = self.crop_y
            img = img[x0:x1, y0:y1, :]
            lab = lab[x0:x1, y0:y1, :]

        # normalize per-volume to 0-1
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        zooms = img_nii.header.get_zooms()[:3]
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        if self.cache_volumes:
            self._cache[cid] = (img, lab, spacing)
        return img, lab, spacing

    def _build_index(self) -> None:
        self.index.clear()
        for i in range(len(self.image_paths)):
            img, lab, _ = self._load_case(i)
            zmax = img.shape[2]

            for z in range(zmax):
                if self.include_empty_slices:
                    self.index.append((i, z))
                else:
                    sl = lab[:, :, z]
                    has_any = np.any(sl == self.root_label) or np.any(
                        sl == self.dura_label
                    )
                    if has_any:
                        self.index.append((i, z))

        if len(self.index) == 0:
            raise RuntimeError(
                "No slices found (index list is empty). include_empty_slices=True にしてください。"
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        case_i, z = self.index[idx]
        cid = self.case_ids[case_i]
        img3d, lab3d, _ = self._load_case(case_i)

        img2d = img3d[:, :, z].astype(np.float32)  # (H,W) ※X,YをH,W扱い
        lab2d = lab3d[:, :, z].astype(np.int16)

        root = (lab2d == self.root_label).astype(np.float32)
        dura = (lab2d == self.dura_label).astype(np.float32)

        img_t = torch.from_numpy(img2d[None, ...])  # (1,H,W)
        mask_t = torch.from_numpy(np.stack([root, dura], 0))  # (2,H,W)

        return img_t, mask_t, cid, z


# =========================================================
# Loss (BCE+Dice) multi-task
# =========================================================
def dice_loss_from_logits_2d(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    # logits/targets: (B,1,H,W)
    probs = torch.sigmoid(logits)
    probs_f = probs.contiguous().view(probs.size(0), -1)
    targ_f = targets.contiguous().view(targets.size(0), -1)
    inter = (probs_f * targ_f).sum(dim=1)
    denom = probs_f.sum(dim=1) + targ_f.sum(dim=1) + eps
    dice = 2.0 * inter / denom
    return 1.0 - dice.mean()


def center_crop_4d_to_match(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # (B,C,H,W)
    _, _, ha, wa = a.shape
    _, _, hb, wb = b.shape
    ht, wt = min(ha, hb), min(wa, wb)

    def crop(x, ht, wt):
        _, _, h, w = x.shape
        hs = (h - ht) // 2
        ws = (w - wt) // 2
        return x[:, :, hs : hs + ht, ws : ws + wt]

    return crop(a, ht, wt), crop(b, ht, wt)


def combined_loss_single_2d(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits, targets = center_crop_4d_to_match(logits, targets)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss_from_logits_2d(logits, targets)
    return bce + dsc


def multitask_loss_2d(
    logits_root: torch.Tensor,
    logits_dura: torch.Tensor,
    masks_2ch: torch.Tensor,  # (B,2,H,W)
    lambda_root: float = 1.0,
    lambda_dura: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root_t = masks_2ch[:, 0:1, ...]
    dura_t = masks_2ch[:, 1:2, ...]
    lr = combined_loss_single_2d(logits_root, root_t)
    ld = combined_loss_single_2d(logits_dura, dura_t)
    total = lambda_root * lr + lambda_dura * ld
    return total, lr, ld


# =========================================================
# Train / Val loops
# =========================================================
def train_one_epoch(
    model, loader, optimizer, device, lambda_root=1.0, lambda_dura=0.3
) -> float:
    model.train()
    run = 0.0
    for imgs, masks, _, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)  # (B,1,H,W)
        masks = masks.to(device)  # (B,2,H,W)

        optimizer.zero_grad()
        log_r, log_d = model(imgs)
        loss, _, _ = multitask_loss_2d(log_r, log_d, masks, lambda_root, lambda_dura)
        loss.backward()
        optimizer.step()
        run += loss.item() * imgs.size(0)
    return run / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, lambda_root=1.0, lambda_dura=0.3) -> float:
    model.eval()
    run = 0.0
    for imgs, masks, _, _ in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        log_r, log_d = model(imgs)
        loss, _, _ = multitask_loss_2d(log_r, log_d, masks, lambda_root, lambda_dura)
        run += loss.item() * imgs.size(0)
    return run / len(loader.dataset)


# =========================================================
# 3D metrics (case-wise) on reconstructed volumes
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
    # distance_transform_edt expects sampling per axis order of array (X,Y,Z)
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

        # reconstruct preds (X,Y,Z)
        X, Y, Z = img3d.shape
        pr_root = np.zeros((X, Y, Z), dtype=np.uint8)
        pr_dura = np.zeros((X, Y, Z), dtype=np.uint8)

        for z in range(Z):
            sl = img3d[:, :, z].astype(np.float32)  # (X,Y)
            inp = torch.from_numpy(sl[None, None, ...]).to(device)  # (1,1,H,W)

            log_r, log_d = model(inp)
            p_r = (torch.sigmoid(log_r)[0, 0].cpu().numpy() > thr_root).astype(np.uint8)
            p_d = (torch.sigmoid(log_d)[0, 0].cpu().numpy() > thr_dura).astype(np.uint8)

            # サイズがズレた場合は center crop で合わせる（保険）
            hx, wx = p_r.shape
            tx, ty = sl.shape
            if (hx, wx) != (tx, ty):
                # center crop/pad to match target (tx,ty)
                out = np.zeros((tx, ty), dtype=np.uint8)
                hs = max((hx - tx) // 2, 0)
                ws = max((wx - ty) // 2, 0)
                hd = max((tx - hx) // 2, 0)
                wd = max((ty - wx) // 2, 0)
                cx = min(hx, tx)
                cy = min(wx, ty)
                out[hd : hd + cx, wd : wd + cy] = p_r[hs : hs + cx, ws : ws + cy]
                p_r = out

                out2 = np.zeros((tx, ty), dtype=np.uint8)
                out2[hd : hd + cx, wd : wd + cy] = p_d[hs : hs + cx, ws : ws + cy]
                p_d = out2

            pr_root[:, :, z] = p_r
            pr_dura[:, :, z] = p_d

        gt_root = (lab3d == root_label).astype(np.uint8)
        gt_dura = (lab3d == dura_label).astype(np.uint8)

        # metrics (3D, spacing in mm)
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
            # embed back to full size if cropped
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

    # CSV
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
# main (train -> test)
# =========================================================
def main():
    p = argparse.ArgumentParser(
        "2D U-Net axial multi-task (root+dura) train -> test -> CSV"
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./ckpt2d_mt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--root_label", type=int, default=1)
    p.add_argument("--dura_label", type=int, default=2)

    p.add_argument("--lambda_root", type=float, default=1.0)
    p.add_argument("--lambda_dura", type=float, default=0.3)

    p.add_argument("--thr_root", type=float, default=0.5)
    p.add_argument("--thr_dura", type=float, default=0.5)

    # ROI crop (optional)
    p.add_argument("--crop_x", type=int, nargs=2, default=None)  # e.g. 50 200
    p.add_argument("--crop_y", type=int, nargs=2, default=None)  # e.g. 45 210

    # slice sampling
    p.add_argument(
        "--include_empty_slices",
        action="store_true",
        help="空スライスも学習に含める（デフォルトは含めない）",
    )

    # save
    p.add_argument("--save_name", type=str, default="best_unet2d_mt.pth")
    p.add_argument(
        "--no_save_nifti", action="store_true", help="テスト推論NIfTIを保存しない"
    )

    # early stopping
    p.add_argument("--patience", type=int, default=30)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- pair Tr / Ts ---
    tr_imgs, tr_labs = pair_paths(args.dataset_root, "Tr")
    ts_imgs, ts_labs = pair_paths(args.dataset_root, "Ts")
    print(f"#Tr cases: {len(tr_imgs)}  #Ts cases: {len(ts_imgs)}")

    # --- split by case ---
    train_imgs, train_labs, val_imgs, val_labs = train_val_split_by_case(
        tr_imgs, tr_labs, val_ratio=args.val_ratio, seed=42
    )
    print(f"#Train cases: {len(train_imgs)}  #Val cases: {len(val_imgs)}")

    crop_x = tuple(args.crop_x) if args.crop_x is not None else None
    crop_y = tuple(args.crop_y) if args.crop_y is not None else None

    # default: “空スライスは含めない” の方が学習が安定しがち
    include_empty = bool(args.include_empty_slices)

    train_ds = AxialSliceDataset(
        train_imgs,
        train_labs,
        root_label=args.root_label,
        dura_label=args.dura_label,
        crop_x=crop_x,
        crop_y=crop_y,
        include_empty_slices=include_empty,
        cache_volumes=True,
    )
    val_ds = AxialSliceDataset(
        val_imgs,
        val_labs,
        root_label=args.root_label,
        dura_label=args.dura_label,
        crop_x=crop_x,
        crop_y=crop_y,
        include_empty_slices=True,  # valは全スライスで loss 見たいので True 推奨
        cache_volumes=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- model ---
    model = UNet2DMultiTask(in_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, args.save_name)

    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_root=args.lambda_root,
            lambda_dura=args.lambda_dura,
        )
        va_loss = validate(
            model,
            val_loader,
            device,
            lambda_root=args.lambda_root,
            lambda_dura=args.lambda_dura,
        )
        print(f"  train_loss: {tr_loss:.4f}  val_loss: {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va_loss,
                    "root_label": args.root_label,
                    "dura_label": args.dura_label,
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                },
                best_path,
            )
            print(f"  >>> Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping: no improvement in val_loss for {args.patience} consecutive epochs."
                )
                break

    # --- load best and test ---
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print("Loaded best checkpoint for test:", best_path)

    pred_out_dir = os.path.join(args.out_dir, "pred_test")
    run_test_casewise(
        model=model,
        ts_imgs=ts_imgs,
        ts_labs=ts_labs,
        out_dir=pred_out_dir,
        root_label=args.root_label,
        dura_label=args.dura_label,
        thr_root=args.thr_root,
        thr_dura=args.thr_dura,
        crop_x=crop_x,
        crop_y=crop_y,
        save_nifti=not args.no_save_nifti,
    )


if __name__ == "__main__":
    main()
