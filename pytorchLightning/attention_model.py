# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import segmentation_models_pytorch as smp  # metrics利用で使う
from monai.networks.nets import AttentionUnet


# ===== 手動設定 =====
LABEL_VALUES = [0, 1, 2]  # 背景, 神経, 硬膜管
WEIGHT_BY_INTENSITY = {
    0: 0.1,  # 背景
    1: 1.5,  # 神経
    2: 0.5,  # 硬膜管
}


def build_class_weights_from_intensity(label_values, weight_map, device):
    """輝度値→クラス重みベクトル (C,) を作る"""
    w = [float(weight_map[v]) for v in label_values]
    return torch.tensor(w, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────
# Weighted Multiclass Dice Loss
# ─────────────────────────────────────────────
class WeightedMulticlassDiceLoss(nn.Module):
    def __init__(self, class_weights=None, eps=1e-6):
        super().__init__()
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            raise TypeError("class_weights must be a torch.Tensor or None")
        self.register_buffer(
            "class_weights",
            class_weights if isinstance(class_weights, torch.Tensor) else None,
        )
        self.eps = eps

    def forward(self, logits, target_idx):
        """
        logits: [B,C,D,H,W]
        target_idx: [B,D,H,W]
        """
        B, C, D, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)  # [B,C,D,H,W]

        # one-hot GT
        target_1h = F.one_hot(target_idx.long(), num_classes=C)  # [B,D,H,W,C]
        target_1h = target_1h.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]

        dims = (0, 2, 3, 4)  # 集計軸
        intersection = torch.sum(probs * target_1h, dim=dims)
        cardinality = torch.sum(probs + target_1h, dim=dims)
        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss_per_class = 1.0 - dice_per_class

        if self.class_weights is None:
            return loss_per_class.mean()

        w = self.class_weights
        if w.numel() != C:
            raise ValueError(f"class_weights has {w.numel()} elems but num_classes={C}")
        return (loss_per_class * w).sum() / (w.sum() + self.eps)


# ─────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────
class MultiClassModel(L.LightningModule):
    """
    3D Attention U-Net (MONAI) によるマルチクラスセグメンテーション
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        label_values=LABEL_VALUES,
        weight_by_intensity=WEIGHT_BY_INTENSITY,
        ce_weight_ratio=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Attention U-Net 3D
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

        # class weights
        class_weights = build_class_weights_from_intensity(
            label_values=self.hparams.label_values,
            weight_map=self.hparams.weight_by_intensity,
            device="cpu",
        )
        self.register_buffer("class_weights", class_weights)

        # 損失関数
        self.dice = WeightedMulticlassDiceLoss(class_weights=self.class_weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean")

        self.ce_weight_ratio = ce_weight_ratio
        self.test_outputs = []

    # --------- utils ---------
    def _ensure_class_indices(self, masks):
        """
        masks をクラスID [B, D, H, W] (Long) に正規化
        """
        if masks.ndim == 4:
            return masks.long()
        if masks.ndim == 5:
            B, C, D, H, W = masks.shape
            if C == 1:
                return masks.squeeze(1).long()
            if C == self.model.out_channels:
                return masks.argmax(dim=1).long()
        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    def _loss(self, logits, masks_idx):
        dice_loss = self.dice(logits, masks_idx)
        ce_loss = self.ce(logits, masks_idx)
        return dice_loss + self.ce_weight_ratio * ce_loss

    # --------- lightning hooks ---------
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks_idx = self._ensure_class_indices(masks)
        logits = self(images)
        loss = self._loss(logits, masks_idx)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks_idx = self._ensure_class_indices(masks)
        logits = self(images)
        loss = self._loss(logits, masks_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        import os, datetime, nibabel as nib
        import numpy as np

        images, masks = batch
        masks_idx = self._ensure_class_indices(masks)
        logits = self(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # 保存ディレクトリ作成
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = f"nifti_predictions-{dt}"
        os.makedirs(out_dir, exist_ok=True)

        for i in range(preds.shape[0]):
            sid = batch_idx * preds.shape[0] + i

            # 予測マスク
            pred_np = preds[i].detach().cpu().numpy().astype(np.uint8)
            nib.save(
                nib.Nifti1Image(pred_np, np.eye(4)),
                os.path.join(out_dir, f"sample_{sid}_pred.nii.gz"),
            )

            # 正解マスク
            gt_np = masks_idx[i].detach().cpu().numpy().astype(np.uint8)
            nib.save(
                nib.Nifti1Image(gt_np, np.eye(4)),
                os.path.join(out_dir, f"sample_{sid}_gt.nii.gz"),
            )

            # 元画像（C=1想定）
            img_np = images[i, 0].detach().cpu().numpy().astype(np.float32)  # [D,H,W]
            nib.save(
                nib.Nifti1Image(img_np, np.eye(4)),
                os.path.join(out_dir, f"sample_{sid}_image.nii.gz"),
            )

        # Dice per-class
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, masks_idx, mode="multiclass", num_classes=logits.shape[1]
        )
        nerve_dice = smp.metrics.f1_score(tp[:, 1], fp[:, 1], fn[:, 1], tn[:, 1]).mean()
        dural_dice = smp.metrics.f1_score(tp[:, 2], fp[:, 2], fn[:, 2], tn[:, 2]).mean()

        self.log("test_dice_nerve", nerve_dice, on_epoch=True, prog_bar=True)
        self.log("test_dice_dural", dural_dice, on_epoch=True, prog_bar=True)
        self.test_outputs.append({"nerve": nerve_dice, "dural": dural_dice})

        return {"nerve": nerve_dice, "dural": dural_dice}

    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("No test outputs collected.")
            return
        nerve = torch.stack([x["nerve"] for x in self.test_outputs]).mean()
        dural = torch.stack([x["dural"] for x in self.test_outputs]).mean()
        print(f"[Test] Nerve Dice: {nerve:.4f} | Dural Dice: {dural:.4f}")
        self.test_outputs = []
