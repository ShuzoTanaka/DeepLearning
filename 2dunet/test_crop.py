import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================
# 3D U-Net（train と同じ構造）
# =====================================


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose3d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)

        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)

    def _center_crop_to(
        self, enc: torch.Tensor, ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, d_ref, h_ref, w_ref = ref.size()
        _, _, d_enc, h_enc, w_enc = enc.size()

        d_target = min(d_ref, d_enc)
        h_target = min(h_ref, h_enc)
        w_target = min(w_ref, w_enc)

        d_start = (d_enc - d_target) // 2
        h_start = (h_enc - h_target) // 2
        w_start = (w_enc - w_target) // 2

        enc_c = enc[
            :,
            :,
            d_start : d_start + d_target,
            h_start : h_start + h_target,
            w_start : w_start + w_target,
        ]

        if (d_ref, h_ref, w_ref) != (d_target, h_target, w_target):
            d_start_r = (d_ref - d_target) // 2
            h_start_r = (h_ref - h_target) // 2
            w_start_r = (w_ref - w_target) // 2
            ref = ref[
                :,
                :,
                d_start_r : d_start_r + d_target,
                h_start_r : h_start_r + h_target,
                w_start_r : w_start_r + w_target,
            ]

        return enc_c, ref

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        e4_c, u4 = self._center_crop_to(e4, u4)
        x4 = torch.cat([u4, e4_c], dim=1)
        d4 = self.dec4(x4)

        u3 = self.up3(d4)
        e3_c, u3 = self._center_crop_to(e3, u3)
        x3 = torch.cat([u3, e3_c], dim=1)
        d3 = self.dec3(x3)

        u2 = self.up2(d3)
        e2_c, u2 = self._center_crop_to(e2, u2)
        x2 = torch.cat([u2, e2_c], dim=1)
        d2 = self.dec2(x2)

        u1 = self.up1(d2)
        e1_c, u1 = self._center_crop_to(e1, u1)
        x1 = torch.cat([u1, e1_c], dim=1)
        d1 = self.dec1(x1)

        out = self.out_conv(d1)
        return out


# ============================
# Dataset（test用・crop込み）
# ============================

CROP_X = (50, 200)
CROP_Y = (45, 210)


class Nifti3DTestDataset(Dataset):
    """
    Test用：画像とラベルをクロップして返す
    """

    def __init__(
        self, image_paths: List[str], label_paths: List[str], nerve_root_label: int = 1
    ):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.nerve_root_label = nerve_root_label
        self.case_ids = [
            os.path.splitext(os.path.basename(p))[0] for p in self.label_paths
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        lab_path = self.label_paths[idx]

        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        img = img_nii.get_fdata().astype(np.float32)  # (X,Y,Z)
        lab = lab_nii.get_fdata().astype(np.int16)

        if img.shape != lab.shape:
            raise ValueError(
                f"Shape mismatch: {img.shape} vs {lab.shape} for {img_path}"
            )

        x_min, x_max = CROP_X
        y_min, y_max = CROP_Y
        if img.shape[0] < x_max or img.shape[1] < y_max:
            raise ValueError(
                f"Volume is smaller than crop region: {img.shape} for {img_path}"
            )

        img = img[x_min:x_max, y_min:y_max, :]  # (Xc,Yc,Z)
        lab = lab[x_min:x_max, y_min:y_max, :]

        # 0-1 正規化（クロップ後）
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        img = img[None, ...]  # (1,Xc,Yc,Z)
        nerve_mask = (lab == self.nerve_root_label).astype(np.float32)  # (Xc,Yc,Z)

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(nerve_mask)
        case_id = self.case_ids[idx]

        # header は返さず、後でパスから再ロードする
        return img_tensor, mask_tensor, case_id, img_path


# ============================
# Dice (test用)
# ============================


def center_crop_3d_to_match(a: np.ndarray, b: np.ndarray):
    """
    a, b: (D,H,W) or (X,Y,Z)
    → 3D を min に合わせて中心クロップして揃える
    """
    assert a.ndim == 3 and b.ndim == 3
    d_a, h_a, w_a = a.shape
    d_b, h_b, w_b = b.shape

    d_t = min(d_a, d_b)
    h_t = min(h_a, h_b)
    w_t = min(w_a, w_b)

    def crop(v, d_t, h_t, w_t):
        d, h, w = v.shape
        d_s = (d - d_t) // 2
        h_s = (h - h_t) // 2
        w_s = (w - w_t) // 2
        return v[d_s : d_s + d_t, h_s : h_s + h_t, w_s : w_s + w_t]

    return crop(a, d_t, h_t, w_t), crop(b, d_t, h_t, w_t)


def dice_coeff_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    pred, target: 3D volume (0/1)
    """
    pred, target = center_crop_3d_to_match(pred, target)

    pred_flat = pred.astype(np.float32).ravel()
    target_flat = target.astype(np.float32).ravel()

    intersection = (pred_flat * target_flat).sum()
    denom = pred_flat.sum() + target_flat.sum() + eps
    return 2.0 * intersection / denom


# ============================
# Utility（パスのペアリング）
# ============================


def pair_ts_paths(dataset_root: str) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(dataset_root, "imagesTs")
    lab_dir = os.path.join(dataset_root, "labelsTs")

    if not os.path.exists(lab_dir):
        raise RuntimeError("labelsTs が存在しません")

    label_files = [
        f for f in os.listdir(lab_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    label_files.sort()

    image_paths = []
    label_paths = []

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


# ============================
# main (推論 + 評価 + 保存)
# ============================


def main():
    parser = argparse.ArgumentParser(description="3D U-Net test (cropped, nerve root)")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--nerve_root_label", type=int, default=1)
    parser.add_argument("--pred_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Test ペア取得
    ts_imgs, ts_labs = pair_ts_paths(args.dataset_root)
    print(f"#Test cases: {len(ts_imgs)}")

    test_ds = Nifti3DTestDataset(
        ts_imgs, ts_labs, nerve_root_label=args.nerve_root_label
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # モデル読み込み
    model = UNet3D(in_channels=1, base_channels=16).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    case_dice: Dict[str, float] = {}

    with torch.no_grad():
        for imgs, gt_nerve, case_ids, img_paths in tqdm(test_loader, desc="Test"):
            imgs = imgs.to(device)  # (1,1,Xc,Yc,Z)

            logits = model(imgs)  # (1,1,Dp,Hp,Wp)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            pred_np = preds.cpu().numpy()[0, 0]  # (Dp,Hp,Wp)
            gt_np = gt_nerve.cpu().numpy()[0]  # (Xc,Yc,Zc)

            dice = dice_coeff_numpy(pred_np, gt_np)
            cid = case_ids[0]
            case_dice[cid] = float(dice)

            print(f"{cid}: Dice = {dice:.4f}")

            # もとの NIfTI から affine / header を取得
            img_path = img_paths[0]
            img_nii = nib.load(img_path)
            affine = img_nii.affine
            header = img_nii.header

            # 予測を NIfTI で保存（クロップ空間）
            pred_nifti = nib.Nifti1Image(pred_np.astype(np.float32), affine, header)
            out_path = os.path.join(args.pred_dir, f"{cid}_pred_nerve.nii.gz")
            nib.save(pred_nifti, out_path)

    print("=== Case-wise Dice (nerve root only, cropped) ===")
    for cid, d in case_dice.items():
        print(f"{cid}: {d:.4f}")
    all_dice = np.array(list(case_dice.values()), dtype=np.float32)
    print(f"Mean Dice (nerve root, cropped): {all_dice.mean():.4f}")


if __name__ == "__main__":
    main()
