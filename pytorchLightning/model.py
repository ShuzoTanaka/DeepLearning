# model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import nibabel as nib
import segmentation_models_pytorch_3d as smp
import datetime

# ===== 手動設定（必要に応じて編集） =====
# マスクの輝度値（順序＝クラス順）
LABEL_VALUES = [0, 1, 2]  # 例: 背景, 神経, 硬膜管

# 輝度値ごとの学習重み（手で調整可能）
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
# 重み付きマルチクラスDice損失（logits入力）
# ─────────────────────────────────────────────
class WeightedMulticlassDiceLoss(nn.Module):
    """
    - 入力: logits [B, C, D, H, W],  target: class indices [B, D, H, W]
    - 処理: Softmax → One-hot → per-class Dice → 重み付き平均
    """

    def __init__(self, class_weights=None, eps=1e-6):
        super().__init__()
        # class_weights は (C,) Tensor を期待
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
        target_idx: [B,D,H,W] (Long)
        """
        B, C, D, H, W = logits.shape
        # Softmax確率
        probs = torch.softmax(logits, dim=1)  # [B,C,D,H,W]
        # One-hot GT
        target_1h = F.one_hot(target_idx.long(), num_classes=C)  # [B,D,H,W,C]
        target_1h = target_1h.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]

        # per-class Dice
        dims = (0, 2, 3, 4)  # 集計軸: batch + 空間
        intersection = torch.sum(probs * target_1h, dim=dims)  # [C]
        cardinality = torch.sum(probs + target_1h, dim=dims)  # [C]
        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss_per_class = 1.0 - dice_per_class  # [C]

        if self.class_weights is None:
            return loss_per_class.mean()
        # 重み付き平均
        w = self.class_weights
        # 安全のためサイズ調整
        if w.numel() != C:
            raise ValueError(f"class_weights has {w.numel()} elems but num_classes={C}")
        return (loss_per_class * w).sum() / (w.sum() + self.eps)


# ─────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────
class MultiClassModel(L.LightningModule):
    """
    3D U-Net (smp-3d) によるマルチクラスセグメンテーション
    - クラス: 背景 / 神経 / 硬膜管（LABEL_VALUESの順）
    - 損失: 重み付きDice + α*重み付きCE
    - masks 形式は以下を許容:
        [B,D,H,W] (class index) / [B,1,D,H,W] (index or intensity) / [B,C,D,H,W] (one-hot)
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        encoder_name="efficientnet-b0",
        label_values=LABEL_VALUES,
        weight_by_intensity=WEIGHT_BY_INTENSITY,
        ce_weight_ratio=0.5,  # 総損失 = Dice + ce_weight_ratio * CE
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=num_classes,
        )

        # 輝度値→クラス重みをBufferとして保持（自動でGPUへ載る）
        class_weights = build_class_weights_from_intensity(
            label_values=self.hparams.label_values,
            weight_map=self.hparams.weight_by_intensity,
            device="cpu",  # Buffer登録は一旦CPUでOK（Lightningが移動）
        )
        self.register_buffer("class_weights", class_weights)

        # 自作: 重み付きマルチクラスDice（logits入力）
        self.dice = WeightedMulticlassDiceLoss(class_weights=self.class_weights)

        # 重み付きCE（logits入力）
        # 注意: torch.nn.CrossEntropyLoss は weight=(C,) Tensor を受け取る
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean")

        self.ce_weight_ratio = ce_weight_ratio
        self.test_outputs = []

    # --------- utils ---------
    def _ensure_class_indices(self, masks):
        """
        masks をクラスID [B, D, H, W] (Long) に正規化:
          - [B,D,H,W] (Long/Int): そのまま
          - [B,1,D,H,W]: squeeze後、輝度集合が label_values に収まるなら intensity→class にマップ
          - [B,C,D,H,W]: one-hot とみなし argmax
        """
        if masks.ndim == 4:
            return masks.long()

        if masks.ndim == 5:
            B, C, D, H, W = masks.shape
            if C == 1:
                m = masks.squeeze(1)  # [B,D,H,W]
                # 輝度→クラスIDへのマッピング（label_valuesがそのままクラスID順）
                uniq = torch.unique(m)
                label_set = torch.tensor(self.hparams.label_values, device=m.device)
                if torch.all(torch.isin(uniq, label_set)):
                    # intensity → class index
                    lut = {
                        int(v): idx for idx, v in enumerate(self.hparams.label_values)
                    }
                    m_np = m.detach().cpu().numpy()
                    mapped = np.vectorize(lambda v: lut.get(int(v), 0))(m_np).astype(
                        np.int64
                    )
                    m = torch.from_numpy(mapped).to(m.device)
                return m.long()

            if C == self.model.segmentation_head.out_channels:
                return masks.argmax(dim=1).long()

        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    def _loss(self, logits, masks_idx):
        # 総損失 = Weighted Dice + α * Weighted CE
        dice_loss = self.dice(logits, masks_idx)  # uses softmax inside
        ce_loss = self.ce(logits, masks_idx)  # CE expects logits + class indices
        return dice_loss + self.ce_weight_ratio * ce_loss

    # --------- lightning hooks ---------
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks_idx = self._ensure_class_indices(masks)  # [B,D,H,W]
        logits = self(images)  # [B,C,D,H,W]
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
        images, masks = batch
        masks_idx = self._ensure_class_indices(masks)  # [B,D,H,W]
        logits = self(images)  # [B,C,D,H,W]
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)  # [B,D,H,W]

        # 保存ディレクトリ作成
        import datetime, os
        import nibabel as nib
        import numpy as np
        import segmentation_models_pytorch as smp

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

            # 元画像（C=1チャンネルを想定）
            img_np = images[i, 0].detach().cpu().numpy().astype(np.float32)  # [D,H,W]
            nib.save(
                nib.Nifti1Image(img_np, np.eye(4)),
                os.path.join(out_dir, f"sample_{sid}_image.nii.gz"),
            )

        # Dice per-class
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, masks_idx, mode="multiclass", num_classes=logits.shape[1]
        )
        nerve_dice = smp.metrics.f1_score(
            tp[:, 1], fp[:, 1], fn[:, 1], tn[:, 1], reduction="none"
        ).mean()
        dural_dice = smp.metrics.f1_score(
            tp[:, 2], fp[:, 2], fn[:, 2], tn[:, 2], reduction="none"
        ).mean()

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
