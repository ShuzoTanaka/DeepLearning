import datetime
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from dataModule import DataModule
from model_2D import MultiClassModel
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    # PyTorchのエラーハンドリング設定
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision("medium")

    # === データモジュールの設定 ===
    dataset_path = "C:/Users/orilab/Desktop/masumoto/MajorityAlgorhythm/pngData"  # データのルートディレクトリ
    split_file = "C:/Users/orilab/Desktop/masumoto/MajorityAlgorhythm/pngData/split/fold_1.txt"  # 交差検証用の分割リスト
    batch_size = 16  # VRAM に応じて調整（4, 8, 16）
    view = "Axial"  # "Axial", "Coronal", "Sagittal" のいずれか

    data_module = DataModule(
        dataset_path=dataset_path,
        split_file=split_file,
        view=view,
        batch_size=batch_size,
        num_workers=4,  # データローダーの並列処理用
    )

    # === モデルの設定 ===
    learning_rate = 1e-3  # 学習率
    model = MultiClassModel(
        in_channels=1, num_classes=3, encoder_name="efficientnet-b0", lr=learning_rate
    )

    print(f"Model type: {type(model)}")
    print(f"Model base classes: {model.__class__.__bases__}")
    print(f"Is instance of LightningModule: {isinstance(model, pl.LightningModule)}")

    # === ログの設定（TensorBoard） ===
    dt = datetime.datetime.now()
    logger = TensorBoardLogger(
        "logs", name=dt.strftime("%Y-%m-%d_%H-%M-%S"), version="version_0"
    )

    # === チェックポイントコールバック ===
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,  # 最も良いモデルのみ保存
        mode="min",
        save_last=True,
    )

    # === トレーナーの設定 ===
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=200,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        precision=16,  # 混合精度学習（VRAM節約）
    )

    print("Starting training...")
    trainer.fit(model, datamodule=data_module)

    # === テストの実行（ランク0のみ） ===
    if trainer.global_rank == 0:
        print("Starting testing...")
        trainer.test(model, datamodule=data_module)
