    """
    Tanaka/PytorchLightning/.lightningenvで実行
    """

from pathlib import Path
import datetime
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from dataModule import DataModule
from model import MultiClassModel
import torch
import pytorch_lightning as pl


# トレーニングスクリプト
if __name__ == "__main__":
    # Torchの精度設定（RTX 4080 Tensorコア対応）
    torch.set_float32_matmul_precision("medium")

    # データセットパス
    dataset_path = Path("data")  # "data/images" と "data/masks" が含まれるフォルダ

    # データモジュールの設定
    data_module = DataModule(dataset_path=dataset_path, batch_size=2)

    # モデルの設定
    model = MultiClassModel(
        in_channels=1,  # 入力チャネル数（グレースケール=1）
        num_classes=3,  # クラス数（背景, nerve, spinal）
        encoder_name="efficientnet-b0"  # エンコーダ
    )

    # ロガーの設定
    dt = datetime.datetime.now()
    logger = TensorBoardLogger(
        "logs", 
        name=dt.strftime("%Y-%m-%d_%H-%M-%S"),  # Windowsで使用可能なフォーマット
        version="version_0"
    )

    # チェックポイントコールバック
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # モニター対象
        dirpath="checkpoints",  # 保存先ディレクトリ
        filename="best-{epoch:02d}-{val_loss:.2f}",  # 保存ファイル名フォーマット
        save_top_k=1,  # 最良モデルのみ保存
        mode="min",  # 低い値が良い
        save_last=True,  # 最終エポックも保存
    )

    # トレーナーの設定
    trainer = L.Trainer(
        accelerator="gpu",  # GPUを使用
        devices=1,  # 使用するGPU数
        logger=logger,  # ロガー
        max_epochs=200,  # 最大エポック数
        callbacks=[checkpoint_callback],  # コールバック
        check_val_every_n_epoch=1,  # 検証間隔
    )

    # トレーニングデータの形状確認（デバッグ用）
    data_module.setup()  # データモジュールを初期化
    train_loader = data_module.train_dataloader()
    for images, masks in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch mask shape: {masks.shape}")
        break  # 最初のバッチのみ確認

    print("Model type:", type(model))  # モデルの型
    print("Model base classes:", model.__class__.__bases__)  # ベースクラス
    print("Is instance of LightningModule:", isinstance(model, pl.LightningModule))

    # トレーニング開始
    trainer.fit(model, datamodule=data_module)

    # テストの実行
    trainer.test(model, datamodule=data_module)

