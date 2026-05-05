    """
    Tanaka/PL/.niseienvで実行
    """

import datetime
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from dataModule_2D import DataModule
from model_2D import MultiClassModel
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = True

    torch.set_float32_matmul_precision("medium")

    dataset_path = "data2"

    # データモジュールの設定
    data_module = DataModule(dataset_path=dataset_path, batch_size=2)

    # モデルの設定
    model = MultiClassModel(in_channels=1, num_classes=3, encoder_name="efficientnet-b0")
    print(type(model))  # モデルの型を確認
    print(isinstance(model, pl.LightningModule))  # LightningModuleかどうか確認


    # ロガーの設定
    dt = datetime.datetime.now()
    logger = TensorBoardLogger(
        "logs", 
        name=dt.strftime("%Y-%m-%d_%H-%M-%S"),
        version="version_0"
    )

    # チェックポイントコールバック
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    # トレーナーの設定
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=50,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
    )
    
    print("Model type:", type(model))  # モデルの型
    print("Model base classes:", model.__class__.__bases__)  # ベースクラス
    print("Is instance of LightningModule:", isinstance(model, pl.LightningModule))

    # トレーニング開始
    trainer.fit(model, datamodule=data_module)

    # テストの実行
    trainer.test(model, datamodule=data_module)
