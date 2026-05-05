import os
import torch
from model_2D import MultiClassModel  # 2D用のモデル定義
from dataModule_2D import DataModule  # 2D用のデータモジュール
import torch.nn.functional as F
from PIL import Image

def save_all_test_data_and_predictions(model, datamodule, output_dir="test_predictions"):
    """
    テストデータの画像、Ground Truth、予測結果をすべて保存する関数。

    Args:
        model (torch.nn.Module): 学習済みモデル。
        datamodule (DataModule): データモジュール。
        output_dir (str): 結果を保存するディレクトリ。
    """
    model.eval()  # モデルを評価モードに設定
    dataloader = datamodule.test_dataloader()

    # 保存先ディレクトリの作成
    image_dir = os.path.join(output_dir, "images")
    true_dir = os.path.join(output_dir, "true_masks")
    pred_dir = os.path.join(output_dir, "pred_masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(true_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    samples_saved = 0  # 保存したサンプル数

    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.cuda(), masks.cuda()  # GPUに転送
        with torch.no_grad():
            predictions = model(images)  # モデルの予測
            predictions = torch.sigmoid(predictions)  # 確率に変換

        for i in range(images.size(0)):
            # 画像、予測マスク、Ground Truthを取得
            image = (images[i].cpu().squeeze().numpy() * 255).astype("uint8")  # 画像
            pred = torch.argmax(predictions[i], dim=0).cpu().numpy().astype("uint8")  # 予測
            true = masks[i].cpu().numpy().astype("uint8")  # Ground Truth

            # 保存
            Image.fromarray(image).save(os.path.join(image_dir, f"sample_{samples_saved}.png"))
            Image.fromarray(pred * 127).save(os.path.join(pred_dir, f"sample_{samples_saved}.png"))  # クラス値を視覚化
            Image.fromarray(true * 127).save(os.path.join(true_dir, f"sample_{samples_saved}.png"))  # Ground Truth

            print(f"Saved sample {samples_saved}:")
            print(f" - Image: {os.path.join(image_dir, f'sample_{samples_saved}.png')}")
            print(f" - Ground Truth: {os.path.join(true_dir, f'sample_{samples_saved}.png')}")
            print(f" - Prediction: {os.path.join(pred_dir, f'sample_{samples_saved}.png')}")

            samples_saved += 1


if __name__ == '__main__':
    # モデルとデータモジュールのインスタンス化
    model_path = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\checkpoints\best-epoch=45-val_loss=0.12.ckpt"
    model = MultiClassModel.load_from_checkpoint(model_path)

    # データモジュールのインスタンス化
    data_module = DataModule(dataset_path="data2", batch_size=1)  # 1枚ずつ評価

    # データモジュールのセットアップ
    data_module.setup()

    # テストデータと予測結果の保存
    save_all_test_data_and_predictions(model, data_module, output_dir="test_predictions")
