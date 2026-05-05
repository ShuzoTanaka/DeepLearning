import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import nibabel as nib
import os
import numpy as np
import lightning as L


class MultiClassModel(L.LightningModule):
    def __init__(self, in_channels=1, num_classes=3, encoder_name="efficientnet-b0"):
        super().__init__()
        # 2D Unetを使用
        self.model = smp.Unet(
            encoder_name=encoder_name,  # エンコーダ（例: efficientnet-b0）
            in_channels=in_channels,  # 入力チャネル数（グレースケール=1）
            classes=num_classes,  # 出力クラス数
        )
        self.loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.test_outputs = []  # test_outputs を初期化

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        images, masks = batch  # masksはクラスインデックス（0, 1, 2）で構成
        predictions = self.model(images)  # 出力形状: [B, 3, H, W]
        loss = self.loss_fn(predictions, masks)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self.model(images)
        loss = self.loss_fn(predictions, masks)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self(images)

        # ワンホットエンコード
        masks = F.one_hot(masks.squeeze(1).long(), num_classes=predictions.shape[1])
        masks = masks.permute(
            0, 4, 1, 2
        ).float()  # [batch_size, num_classes, height, width]

        # バイナリマスクに変換（閾値0.5）し、適切な型にキャスト
        binary_predictions = (
            (predictions.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
        )

        # 各クラスのマスクを保存
        output_dir = "nifti_predictions1210_2"
        os.makedirs(output_dir, exist_ok=True)
        for i in range(predictions.shape[0]):  # バッチ内の各サンプル
            print("in batches")
            sample_idx = batch_idx * predictions.shape[0] + i

            # 1. 元の画像データを保存
            image_data = images[i].cpu().numpy().squeeze()  # Shape: [H, W]
            image_path = os.path.join(output_dir, f"sample_{sample_idx}_image.nii.gz")
            nib.save(nib.Nifti1Image(image_data, np.eye(4)), image_path)

            # 2. Ground Truth（マスク）を保存
            ground_truth = masks[i].cpu().numpy().astype(np.uint8)  # Shape: [C, H, W]
            for class_idx in range(ground_truth.shape[0]):  # 各クラス
                gt_class_mask = ground_truth[class_idx]  # Shape: [H, W]
                gt_path = os.path.join(
                    output_dir, f"sample_{sample_idx}_class_{class_idx}_gt.nii.gz"
                )
                nib.save(nib.Nifti1Image(gt_class_mask, np.eye(4)), gt_path)

            # 3. 予測マスクを保存
            for class_idx in range(predictions.shape[1]):  # 各クラス
                pred_class_mask = binary_predictions[i, class_idx]  # Shape: [H, W]
                pred_path = os.path.join(
                    output_dir, f"sample_{sample_idx}_class_{class_idx}_pred.nii.gz"
                )
                nib.save(nib.Nifti1Image(pred_class_mask, np.eye(4)), pred_path)

        # Dice係数の計算
        tp, fp, fn, tn = smp.metrics.get_stats(
            predictions.sigmoid() > 0.5,  # Threshold predictions
            masks.int(),
            mode="multiclass",
            num_classes=predictions.shape[1],
        )

        # 各クラスのDice係数（特にクラス1）を計算
        class1_dice = smp.metrics.f1_score(
            tp[:, 1], fp[:, 1], fn[:, 1], tn[:, 1], reduction="none"
        ).mean()  # バッチ全体の平均を計算

        # 保存用のリストに追加
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
