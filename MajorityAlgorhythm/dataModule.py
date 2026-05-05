from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PNGDataset
import lightning as L


class DataModule(L.LightningDataModule):
    def __init__(
        self, dataset_path, split_file, view="Axial", batch_size=16, num_workers=4
    ):
        """
        PyTorch Lightning DataModule for loading PNG datasets.

        Args:
            dataset_path (str): Path to the dataset directory.
            split_file (str): Path to the split file (train/val/test).
            view (str): "Axial", "Coronal", or "Sagittal" indicating the dataset view.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of worker threads for DataLoader.
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.split_file = Path(split_file)
        self.view = view
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Prepare datasets for training, validation, and testing.
        """
        print(f"Preparing data for {self.view} view...")

        # `splits/fold_1.txt` から train, val, test を取得
        with open(self.split_file, "r") as f:
            lines = f.readlines()
            train_cases = lines[0].strip().split(":")[1].split(", ")
            val_cases = lines[1].strip().split(":")[1].split(", ")
            test_cases = lines[2].strip().split(":")[1].split(", ")

        def get_case_paths(cases, folder):
            return [folder / case / self.view for case in cases]

        # 画像とマスクのパスリストを取得
        train_image_dirs = get_case_paths(train_cases, self.dataset_path / "images")
        train_mask_dirs = get_case_paths(train_cases, self.dataset_path / "masks")

        val_image_dirs = get_case_paths(val_cases, self.dataset_path / "images")
        val_mask_dirs = get_case_paths(val_cases, self.dataset_path / "masks")

        test_image_dirs = get_case_paths(test_cases, self.dataset_path / "images")
        test_mask_dirs = get_case_paths(test_cases, self.dataset_path / "masks")

        # データセットの作成
        self.train_dataset = PNGDataset(train_image_dirs, train_mask_dirs)
        self.val_dataset = PNGDataset(val_image_dirs, val_mask_dirs)
        self.test_dataset = PNGDataset(test_image_dirs, test_mask_dirs)

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


if __name__ == "__main__":
    # データモジュールの動作確認
    dataset_path = r"C:/Users/orilab/Desktop/masumoto/MajorityAlgorhythm/pngData"
    split_file = (
        r"C:/Users/orilab/Desktop/masumoto/MajorityAlgorhythm/pngData/split/fold_1.txt"
    )
    batch_size = 16
    view = "Axial"  # "Coronal" または "Sagittal" に変更して確認可能

    dm = DataModule(dataset_path, split_file, view, batch_size)
    dm.setup()

    print("Train DataLoader:")
    for images, masks in dm.train_dataloader():
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break

    print("Validation DataLoader:")
    for images, masks in dm.val_dataloader():
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break

    print("Test DataLoader:")
    for images, masks in dm.test_dataloader():
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break
