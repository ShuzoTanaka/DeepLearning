    """
    保存がpng version
    """

import os
import torch
from lightning.pytorch import Trainer
from model_2D import MultiClassModel  # 2D用のモデル定義
from dataModule_2D import DataModule  # 2D用のデータモジュール
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from PIL import Image


def calculate_class_dice_score_and_save_samples(
    model, datamodule, target_class, output_dir="predictions", num_samples=5
):
    """
    特定のクラスに対するDiceスコアを計算し、初めのサンプルを保存する関数。

    Args:
        model (torch.nn.Module): 学習済みモデル。
        datamodule (DataModule): データモジュール。
        target_class (int): Diceスコアを計算する対象クラス。
        output_dir (str): 予測結果を保存するディレクトリ。
        num_samples (int): 保存するサンプル数。
    """
    model.eval()  # モデルを評価モードに設定
    dataloader = datamodule.test_dataloader()

    # 保存先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    dice_scores = []
    samples_saved = 0  # 保存したサンプル数

    for batch_idx, (images, masks) in enumerate(dataloader):
        if samples_saved >= num_samples:
            break

        images, masks = images.cuda(), masks.cuda()  # GPUに転送
        with torch.no_grad():
            predictions = model(images)  # モデルの予測
            predictions = torch.sigmoid(predictions)  # 確率に変換

        # 特定クラスだけを抽出
        pred_class = predictions[:, target_class, :, :]  # 対象クラスの予測 [B, H, W]
        true_class = (masks == target_class).long()  # 対象クラスのマスク [B, H, W]

        # Ground Truthがすべてゼロの場合はスキップ
        if torch.sum(true_class) == 0:
            print(
                f"Skipping batch {batch_idx} as ground truth is all zeros for class {target_class}."
            )
            continue

        # Diceスコアの計算
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_class > 0.5,  # バイナリ化
            true_class,
            mode="binary",  # バイナリモードで計算
        )
        dice_score = smp.metrics.f1_score(
            tp, fp, fn, tn, reduction="none"
        ).mean()  # バッチ内の平均を計算
        dice_scores.append(dice_score.item())

        # 保存するサンプルを準備
        image = (images[0].cpu().squeeze().numpy() * 255).astype("uint8")
        pred = (pred_class[0].cpu().numpy() * 255).astype("uint8")
        true = (true_class[0].cpu().numpy() * 255).astype("uint8")

        # 保存
        Image.fromarray(image).save(
            os.path.join(output_dir, f"sample_{samples_saved}_image.png")
        )
        Image.fromarray(pred).save(
            os.path.join(output_dir, f"sample_{samples_saved}_pred.png")
        )
        Image.fromarray(true).save(
            os.path.join(output_dir, f"sample_{samples_saved}_true.png")
        )

        print(f"Saved sample {samples_saved}: Dice Score = {dice_score:.4f}")

        samples_saved += 1

    # 平均Diceスコアを計算
    if dice_scores:
        mean_dice = sum(dice_scores) / len(dice_scores)
        print(f"Average Dice Score for class {target_class}: {mean_dice:.4f}")
    else:
        print(
            f"No valid samples for class {target_class}. Dice score could not be calculated."
        )


if __name__ == "__main__":
    # モデルとデータモジュールのインスタンス化
    model_path = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\checkpoints\best-epoch=45-val_loss=0.12.ckpt"
    model = MultiClassModel.load_from_checkpoint(model_path)

    # データモジュールのインスタンス化
    data_module = DataModule(dataset_path="data4", batch_size=1)  # 1枚ずつ評価

    # データモジュールのセットアップ
    data_module.setup()

    # Diceスコアの計算と出力、サンプルの保存（クラス1を指定）
    calculate_class_dice_score_and_save_samples(
        model,
        data_module,
        target_class=1,
        output_dir="predictions_1210_2d",
        num_samples=50,
    )
