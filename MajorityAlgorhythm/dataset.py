import os
import glob
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np


class PNGDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None):
        """
        Dataset for 2D segmentation tasks using PNG files.

        Args:
            image_dirs (list of Path or str): List of directories containing the image PNG files.
            mask_dirs (list of Path or str): List of directories containing the mask PNG files.
            transform (callable, optional): Optional transform to be applied on an image or mask.
        """
        self.image_files = []
        self.mask_files = []

        # すべてのディレクトリからPNGファイルを収集
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            image_dir = str(image_dir)  # Pathオブジェクトを文字列に変換
            mask_dir = str(mask_dir)

            self.image_files.extend(sorted(glob.glob(os.path.join(image_dir, "*.png"))))
            self.mask_files.extend(sorted(glob.glob(os.path.join(mask_dir, "*.png"))))

        assert len(self.image_files) == len(
            self.mask_files
        ), "Images and masks count do not match!"
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像とマスクのパスを取得
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # 画像とマスクを読み込む（グレースケール）
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # マスクをnumpy配列に変換
        mask = np.array(mask)

        # # リマップ処理
        # mask = np.where(mask == 127, 1, mask)
        # mask = np.where(mask == 255, 2, mask)
        # mask = np.where(mask == 0, 0, mask)

        # mask のリマッピング (背景: 0, クラス1: 1, クラス2: 2)
        mask = np.array(mask)
        mask = np.where(mask == 127, 1, mask)  # 127 を 1 に変換
        mask = np.where(mask == 255, 2, mask)  # 255 を 2 に変換
        mask = np.where(mask > 2, 2, mask)  # それ以外の異常値は 2 に変換
        mask = np.where(mask < 0, 0, mask)  # 負の値があれば 0 に変換

        # トランスフォームを適用（必要なら）
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # テンソルに変換
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(
            0
        )  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.long)  # [H, W]

        return image, mask


def main():
    # データセットディレクトリ
    base_dir = r"C:/Users/orilab/Desktop/masumoto/MajorityAlgorhythm/pngData"
    image_dirs = [os.path.join(base_dir, "images", "00001", "Axial")]  # テスト用に1症例
    mask_dirs = [os.path.join(base_dir, "masks", "00001", "Axial")]

    # データセットの作成
    dataset = PNGDataset(image_dirs=image_dirs, mask_dirs=mask_dirs)

    # サンプルの確認
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        print(f"Sample {idx + 1}:")
        print(f"  Image shape: {image.shape}")  # [1, H, W]
        print(f"  Mask shape: {mask.shape}")  # [H, W]
        print(f"  Mask unique values: {torch.unique(mask)}")

        if idx >= 5:  # 最初の5サンプルだけ確認
            break

    print("Dataset loaded successfully.")


if __name__ == "__main__":
    main()
