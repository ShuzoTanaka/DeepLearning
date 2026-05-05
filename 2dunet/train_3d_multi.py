import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================
# 3D U-Net (Encoder + shared decoder)
# ============================


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


class MultiTaskUNet3D(nn.Module):
    """
    3Dマルチタスク U-Net
      - Encoder〜低解像度側 Decoder を共有
      - 高解像度側 Decoder を 神経根 / 硬膜管 で分岐
    出力:
      out_root: 神経根 (B,1,D,H,W)
      out_dura: 硬膜管 (B,1,D,H,W)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)

        # shared decoder (coarse)
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

        # root branch (fine)
        self.up1_root = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1_root = DoubleConv3D(base_channels * 2, base_channels)
        self.out_root = nn.Conv3d(base_channels, 1, kernel_size=1)

        # dura branch (fine)
        self.up1_dura = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1_dura = DoubleConv3D(base_channels * 2, base_channels)
        self.out_dura = nn.Conv3d(base_channels, 1, kernel_size=1)

    def _center_crop_to(
        self, enc: torch.Tensor, ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        enc を ref と同じ (D,H,W) に中心クロップし、
        必要なら ref 側も同じサイズに中心クロップして返す。
        """
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # shared decoder
        u4 = self.up4(b)
        e4_c, u4 = self._center_crop_to(e4, u4)
        d4 = self.dec4(torch.cat([u4, e4_c], dim=1))

        u3 = self.up3(d4)
        e3_c, u3 = self._center_crop_to(e3, u3)
        d3 = self.dec3(torch.cat([u3, e3_c], dim=1))

        u2 = self.up2(d3)
        e2_c, u2 = self._center_crop_to(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2_c], dim=1))

        # root branch
        u1r = self.up1_root(d2)
        e1_cr, u1r = self._center_crop_to(e1, u1r)
        d1r = self.dec1_root(torch.cat([u1r, e1_cr], dim=1))
        out_root = self.out_root(d1r)

        # dura branch
        u1d = self.up1_dura(d2)
        e1_cd, u1d = self._center_crop_to(e1, u1d)
        d1d = self.dec1_dura(torch.cat([u1d, e1_cd], dim=1))
        out_dura = self.out_dura(d1d)

        return out_root, out_dura


# ============================
# Dataset (3D volume / multi-task)
# ============================


class Nifti3DDataset(Dataset):
    """
    1症例 = 1サンプル（3D volumeそのまま）。
    マスクは2チャネル:
      ch0: 神経根 (nerve_root_label == 1など)
      ch1: 硬膜管 (dura_label == 2など)
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        nerve_root_label: int = 1,
        dura_label: int = 2,
    ):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.nerve_root_label = nerve_root_label
        self.dura_label = dura_label

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

        # ====== ★ 固定範囲クロップをここで実施（あなたの指定通り） ======
        # 50 < x < 200  → Python index 50:200
        # 45 < y < 210  → Python index 45:210
        # z は全範囲
        img = img[50:200, 45:210, :]
        lab = lab[50:200, 45:210, :]
        # ===========================================================

        # 強度正規化 (0-1)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        img = img[None, ...]  # (1, X, Y, Z)

        # マルチタスク用マスク
        root_mask = (lab == self.nerve_root_label).astype(np.float32)
        dura_mask = (lab == self.dura_label).astype(np.float32)

        mask = np.stack([root_mask, dura_mask], axis=0)  # (2,X,Y,Z)

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        case_id = self.case_ids[idx]
        return img_tensor, mask_tensor, case_id


# ============================
# Loss & Dice
# ============================


def center_crop_5d_to_match(a: torch.Tensor, b: torch.Tensor):
    """
    a, b: (B, C, D, H, W)
    → D,H,W を min に合わせて中心クロップして揃える
    """
    assert a.dim() == 5 and b.dim() == 5
    _, _, d_a, h_a, w_a = a.shape
    _, _, d_b, h_b, w_b = b.shape

    d_t = min(d_a, d_b)
    h_t = min(h_a, h_b)
    w_t = min(w_a, w_b)

    def crop(t, d_t, h_t, w_t):
        _, _, d, h, w = t.shape
        d_s = (d - d_t) // 2
        h_s = (h - h_t) // 2
        w_s = (w - w_t) // 2
        return t[:, :, d_s : d_s + d_t, h_s : h_s + h_t, w_s : w_s + w_t]

    a_c = crop(a, d_t, h_t, w_t)
    b_c = crop(b, d_t, h_t, w_t)
    return a_c, b_c


def center_crop_3d_to_match(a: np.ndarray, b: np.ndarray):
    """
    a, b: (D,H,W)
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


def dice_loss_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    logits: (B,1,D,H,W)
    targets: (B,1,D,H,W)
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.contiguous().view(probs.size(0), -1)
    targets_flat = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps
    dice = 2.0 * intersection / denom
    return 1.0 - dice.mean()


def combined_loss_single(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    単一タスク用の BCE + Dice
    (logits, targets ともに (B,1,D,H,W))
    """
    logits_aligned, targets_aligned = center_crop_5d_to_match(logits, targets)

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits_aligned, targets_aligned
    )
    dsc = dice_loss_from_logits(logits_aligned, targets_aligned)
    return bce + dsc


def multitask_loss(
    logits_root: torch.Tensor,
    logits_dura: torch.Tensor,
    masks: torch.Tensor,
    lambda_root: float = 1.0,
    lambda_dura: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    masks: (B,2,D,H,W)
      ch0: root, ch1: dura
    """
    root_targets = masks[:, 0:1, ...]  # (B,1,D,H,W)
    dura_targets = masks[:, 1:2, ...]  # (B,1,D,H,W)

    loss_root = combined_loss_single(logits_root, root_targets)
    loss_dura = combined_loss_single(logits_dura, dura_targets)

    loss = lambda_root * loss_root + lambda_dura * loss_dura
    return loss, loss_root, loss_dura


def dice_coeff_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred, target = center_crop_3d_to_match(pred, target)

    pred_flat = pred.astype(np.float32).ravel()
    target_flat = target.astype(np.float32).ravel()
    intersection = (pred_flat * target_flat).sum()
    denom = pred_flat.sum() + target_flat.sum() + eps
    return 2.0 * intersection / denom


# ============================
# Train / Val / Test loops
# ============================


def train_one_epoch(model, loader, optimizer, device, lambda_root=1.0, lambda_dura=0.3):
    model.train()
    running_loss = 0.0

    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)  # (B,1,D,H,W)
        masks = masks.to(device)  # (B,2,D,H,W)

        optimizer.zero_grad()
        logits_root, logits_dura = model(imgs)
        loss, loss_root, loss_dura = multitask_loss(
            logits_root,
            logits_dura,
            masks,
            lambda_root=lambda_root,
            lambda_dura=lambda_dura,
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate(model, loader, device, lambda_root=1.0, lambda_dura=0.3):
    model.eval()
    running_loss = 0.0

    for imgs, masks, _ in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits_root, logits_dura = model(imgs)
        loss, loss_root, loss_dura = multitask_loss(
            logits_root,
            logits_dura,
            masks,
            lambda_root=lambda_root,
            lambda_dura=lambda_dura,
        )
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate_casewise_nerve_dice(model, loader: DataLoader, device) -> Dict[str, float]:
    """
    神経根 Dice を症例ごとに計算
    （硬膜管は任意で拡張可）
    """
    model.eval()
    case_dice: Dict[str, float] = {}

    for imgs, masks, case_ids in tqdm(loader, desc="Test", leave=False):
        imgs = imgs.to(device)  # (1,1,D,H,W)
        masks = masks.to(device)  # (1,2,D,H,W)

        logits_root, logits_dura = model(imgs)
        probs_root = torch.sigmoid(logits_root)
        preds_root = (probs_root > 0.5).float()

        preds_np = preds_root.cpu().numpy()[0, 0]  # (D,H,W)
        masks_np = masks.cpu().numpy()[0, 0]  # ch0: root

        dice = dice_coeff_numpy(preds_np, masks_np)
        cid = case_ids[0]
        case_dice[cid] = float(dice)

    print("=== Case-wise Dice (nerve root only) ===")
    for cid, d in case_dice.items():
        print(f"{cid}: {d:.4f}")

    all_dice = np.array(list(case_dice.values()), dtype=np.float32)
    print(f"Mean Dice (nerve root): {all_dice.mean():.4f}")

    return case_dice


# ============================
# Utility
# ============================


def pair_tr_paths(dataset_root: str) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(dataset_root, "imagesTr")
    lab_dir = os.path.join(dataset_root, "labelsTr")

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
                f"Image not found for label {lf}: expected {img_path}"
            )

        image_paths.append(img_path)
        label_paths.append(lab_path)

    return image_paths, label_paths


def pair_ts_paths(dataset_root: str) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(dataset_root, "imagesTs")
    lab_dir = os.path.join(dataset_root, "labelsTs")

    if not os.path.exists(lab_dir):
        raise RuntimeError(
            "labelsTs が存在しません（テスト用ラベルが必要な場合は作成してください）"
        )

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


def train_val_split(
    image_paths: List[str],
    label_paths: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
):
    assert len(image_paths) == len(label_paths)
    n = len(image_paths)
    indices = list(range(n))

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def subset(lst, idxs):
        return [lst[i] for i in idxs]

    train_imgs = subset(image_paths, train_idx)
    train_labs = subset(label_paths, train_idx)
    val_imgs = subset(image_paths, val_idx)
    val_labs = subset(label_paths, val_idx)

    return train_imgs, train_labs, val_imgs, val_labs


# ============================
# main
# ============================


def main():
    parser = argparse.ArgumentParser(description="3D Multi-task U-Net (root + dura)")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset フォルダ (直下に imagesTr, imagesTs, labelsTr, labelsTs)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--batch_size", type=int, default=1, help="3D volume なので通常1"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--nerve_root_label", type=int, default=1, help="神経根ラベル値"
    )
    parser.add_argument("--dura_label", type=int, default=2, help="硬膜管ラベル値")
    parser.add_argument("--out_dir", type=str, default="./ckpt3d_mt")
    parser.add_argument("--num_workers", type=int, default=2)

    # loss weight
    parser.add_argument("--lambda_root", type=float, default=1.0)
    parser.add_argument("--lambda_dura", type=float, default=0.3)

    parser.add_argument(
        "--save_name",
        type=str,
        default="best_3dunet",
        help="保存する .pth のファイル名（拡張子は自動で .pth）",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Tr
    tr_imgs, tr_labs = pair_tr_paths(args.dataset_root)
    print(f"#Total Tr cases: {len(tr_imgs)}")

    train_imgs, train_labs, val_imgs, val_labs = train_val_split(
        tr_imgs, tr_labs, val_ratio=args.val_ratio, seed=42
    )
    print(f"#Train cases: {len(train_imgs)}, #Val cases: {len(val_imgs)}")

    # Ts
    ts_imgs, ts_labs = pair_ts_paths(args.dataset_root)
    print(f"#Test cases: {len(ts_imgs)}")

    # Dataset / DataLoader
    train_ds = Nifti3DDataset(
        train_imgs,
        train_labs,
        nerve_root_label=args.nerve_root_label,
        dura_label=args.dura_label,
    )
    val_ds = Nifti3DDataset(
        val_imgs,
        val_labs,
        nerve_root_label=args.nerve_root_label,
        dura_label=args.dura_label,
    )
    test_ds = Nifti3DDataset(
        ts_imgs,
        ts_labs,
        nerve_root_label=args.nerve_root_label,
        dura_label=args.dura_label,
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
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = MultiTaskUNet3D(in_channels=1, base_channels=16)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, f"{args.save_name}.pth")

    patience = 30
    epochs_no_improve = 0

    # train loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_root=args.lambda_root,
            lambda_dura=args.lambda_dura,
        )
        val_loss = validate(
            model,
            val_loader,
            device,
            lambda_root=args.lambda_root,
            lambda_dura=args.lambda_dura,
        )
        print(f"  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  >>> Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping: no improvement in val_loss for {patience} consecutive epochs."
                )
                break

    # test
    if os.path.exists(best_path):
        print("Loading best checkpoint for test...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    _case_dice = evaluate_casewise_nerve_dice(model, test_loader, device)


if __name__ == "__main__":
    main()
