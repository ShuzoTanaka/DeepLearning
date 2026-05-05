import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np

class PNGDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Dataset for 2D segmentation tasks using PNG files.

        Args:
            image_dir (str): Path to the directory containing the image PNG files.
            mask_dir (str): Path to the directory containing the mask PNG files.
            transform (callable, optional): Optional transform to be applied
                on an image or mask.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))  # PNGファイルをソートして取得
        self.mask_files = sorted(os.listdir(mask_dir))  # PNGファイルをソートして取得
        self.transform = transform
        assert len(self.image_files) == len(self.mask_files), "Images and masks count do not match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像とマスクのパスを取得
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # 画像とマスクを読み込む
        image = Image.open(image_path).convert("L")  # グレースケールとして読み込み
        mask = Image.open(mask_path).convert("L")  # グレースケールとして読み込み

        # マスクをnumpy配列に変換
        mask = np.array(mask)
        
        # リマップ処理
        mask = np.where(mask == 127, 1, mask)
        mask = np.where(mask == 255, 2, mask)
        mask = np.where(mask == 0, 0, mask)

        # トランスフォームを適用（必要なら）
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # テンソルに変換
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.long)  # [H, W]

        return image, mask

def main():
    # データセットディレクトリ
    base_dir = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\data2"
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    # データセットの作成
    dataset = PNGDataset(image_dir=image_dir, mask_dir=mask_dir)

    # サンプルの確認
    for idx in range(len(dataset)):  # 全サンプルを確認
        image, mask = dataset[idx]
        print(f"Sample {idx + 1}:")
        print(f"  Image shape: {image.shape}")  # [1, H, W]
        print(f"  Mask shape: {mask.shape}")  # [H, W]
        print(f"  Mask unique values: {torch.unique(mask)}")

        if idx >= 15:  # 最初の5サンプルだけ確認
            break

    print("Dataset loaded successfully.")

if __name__ == "__main__":
    main()
