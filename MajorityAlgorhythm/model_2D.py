import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import nibabel as nib
import os
import numpy as np
import lightning as L


class MultiClassModel(L.LightningModule):
    def __init__(
        self, in_channels=1, num_classes=3, encoder_name="efficientnet-b0", lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        # U-Net モデル
        self.model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            encoder_weights="imagenet",
            classes=num_classes,
        )

        # **複合損失関数を明示的に定義**
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.lr = lr
        self.test_outputs = []

    def forward(self, x):

        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_loss(self, predictions, masks):
        """
        Dice Loss + CrossEntropy Loss の組み合わせ
        """
        # mask の値を 0,1 に変換
        masks = torch.where(masks == 2, 1, masks)

        dice = self.dice_loss(predictions, masks)
        ce = self.ce_loss(predictions, masks)
        return dice + ce

    def training_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self.model(images)  # 出力: [B, num_classes, H, W]

        loss = self.compute_loss(predictions, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self.model(images)

        loss = self.compute_loss(predictions, masks)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self(images)  # 出力: [B, num_classes, H, W]

        # 確率に変換（Softmax）
        probs = torch.softmax(predictions, dim=1)
        pred_classes = torch.argmax(probs, dim=1)  # Shape: [B, H, W]

        # One-hot エンコード（GTマスク）
        masks_one_hot = F.one_hot(masks.long(), num_classes=predictions.shape[1])
        masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # NIfTI で保存
        output_dir = "nifti_predictions0211"
        os.makedirs(output_dir, exist_ok=True)

        for i in range(predictions.shape[0]):  # バッチ内の各サンプル
            sample_idx = batch_idx * predictions.shape[0] + i

            # 1. 画像データを保存
            image_data = images[i].cpu().numpy().squeeze()
            image_path = os.path.join(output_dir, f"sample_{sample_idx}_image.nii.gz")
            nib.save(nib.Nifti1Image(image_data, np.eye(4)), image_path)

            # 2. Ground Truth（マスク）を保存
            gt_class_mask = masks[i].cpu().numpy().astype(np.uint8)
            gt_path = os.path.join(output_dir, f"sample_{sample_idx}_gt.nii.gz")
            nib.save(nib.Nifti1Image(gt_class_mask, np.eye(4)), gt_path)

            # 3. 予測マスクを保存
            pred_class_mask = pred_classes[i].cpu().numpy().astype(np.uint8)
            pred_path = os.path.join(output_dir, f"sample_{sample_idx}_pred.nii.gz")
            nib.save(nib.Nifti1Image(pred_class_mask, np.eye(4)), pred_path)

        # Dice係数の計算
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_classes,
            masks.int(),
            mode="multiclass",
            num_classes=predictions.shape[1],
        )

        # 各クラスの Dice 計算
        dice_scores = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        class1_dice = dice_scores[1].mean()  # クラス1の平均 Dice

        self.test_outputs.append({"test_dice_score": class1_dice})

        # ログ出力
        self.log("test_dice_score", class1_dice, on_epoch=True)
        return {"test_dice_score": class1_dice}

    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("Error: self.test_outputs is empty. Check test_step implementation.")
            return

        all_dice_scores = torch.stack([x["test_dice_score"] for x in self.test_outputs])
        mean_dice_scores = all_dice_scores.mean(dim=0)

        if mean_dice_scores.ndim > 0:
            for class_idx, dice_score in enumerate(mean_dice_scores):
                print(f"Class {class_idx} Dice coefficient: {dice_score.item():.4f}")
        else:
            print(
                "Error: mean_dice_scores is not iterable. Check Dice score calculation."
            )

        overall_dice = mean_dice_scores.mean()
        print(f"Overall Dice coefficient: {overall_dice.item():.4f}")

        self.test_outputs = []
