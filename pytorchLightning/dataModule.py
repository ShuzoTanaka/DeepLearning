from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from dataset import NiftiDataset
from monai.transforms import Compose, RandFlipd, RandRotate90d, ToTensord
from monai.transforms import RandGaussianNoised


class DataModule(L.LightningDataModule):
    def __init__(self, dataset_path, split_file, batch_size=16, num_workers=0):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.split_file = Path(split_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = Compose(
            [
                RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
                RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
                ToTensord(keys=["image", "mask"]),
            ]
        )
        self.test_transform = None  # テストでは拡張なし

    def _load_splits(self):
        """
        指定されたfold.txtファイルから症例リストを読み込む。
        """
        splits = {"train": [], "val": [], "test": []}
        with open(self.split_file, "r") as f:
            for line in f.readlines():
                key, values = line.strip().split(":")
                splits[key.strip()] = [v.strip() for v in values.split(",")]
        return splits

    def setup(self, stage=None):
        print("Preparing data based on predefined splits...")
        image_folder = self.dataset_path / "images"
        mask_folder = self.dataset_path / "masks"

        # 訓練・検証・テストデータの症例リストをロード
        splits = self._load_splits()
        train_cases = splits["train"]
        val_cases = splits["val"]
        test_cases = splits["test"]

        # データセットを作成
        self.train_dataset = NiftiDataset(
            image_folder,
            mask_folder,
            case_list=train_cases,
            transform=self.train_transform,
        )
        self.val_dataset = NiftiDataset(
            image_folder,
            mask_folder,
            case_list=val_cases,
            transform=self.test_transform,
        )
        self.test_dataset = NiftiDataset(
            image_folder,
            mask_folder,
            case_list=test_cases,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
