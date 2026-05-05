# test.py
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from lightning.pytorch import Trainer
from attention_model import MultiClassModel
from dataModule import DataModule
from pathlib import Path


def main():
    torch.set_float32_matmul_precision("medium")  # Tensor Cores を有効活用（推奨）

    model_path = "C:/Users/orilab/Desktop/masumoto/pytorchLightning/checkpoints/best-2025-09-18_20-50-29-epoch=299-val_loss=1.16.ckpt"
    model = MultiClassModel.load_from_checkpoint(model_path)

    split_path = Path("data/split/fold_1.txt")

    # ★ デバッグのため test 時は num_workers=0 を強制（下で実例）
    dm = DataModule(
        dataset_path="data", batch_size=8, split_file=split_path, num_workers=0
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        # fast_dev_run=1,        # ← まずは配線確認したい時に便利
        # num_sanity_val_steps=0 # datamoduleの初期化が重いときは有効
    )
    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    # Windows では spawn 前提。明示的に指定しておくと安心
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
