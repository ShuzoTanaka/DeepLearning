from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np


class NiftiDataset(Dataset):
    def __init__(self, image_folder, mask_folder, target_shape=(256, 256, 64)):
        """
        NiftiDataset for 3D segmentation tasks.
        
        Args:
            image_folder (str or Path): Path to the folder containing input images (.nii or .nii.gz).
            mask_folder (str or Path): Path to the folder containing segmentation masks (.nii or .nii.gz).
            target_shape (tuple): Target shape (H, W, D) to which all images and masks will be resized.
        """
        self.image_paths = sorted(Path(image_folder).glob("*.nii"))
        self.mask_paths = sorted(Path(mask_folder).glob("*.nii.gz"))
        self.target_shape = target_shape  # Target shape for resizing
        
        assert len(self.image_paths) == len(self.mask_paths), \
            "The number of images and masks must be the same!"
        
        # Ensure matching filenames
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            assert img_path.name.split('.')[0] == mask_path.name.split('.')[0], \
                f"Image {img_path} and mask {mask_path} names do not match!"


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and preprocess a single sample from the dataset.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            image (torch.Tensor): Preprocessed input image tensor of shape (1, H, W, D).
            mask (torch.Tensor): Preprocessed mask tensor of shape (1, H, W, D).
        """
        # Load image and mask using nibabel
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Normalize image and add channel dimension
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        

        # Convert mask to tensor and add channel dimension
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)


        # Resize image and mask to the target shape
        image = F.interpolate(image.unsqueeze(0), size=self.target_shape, mode="trilinear", align_corners=False).squeeze(0)

        # 一時的に torch.float32 にキャストしてリサイズ
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.target_shape, mode="nearest").squeeze(0)

        # 再度 torch.long にキャスト
        mask = mask.long()


        return image, mask


if __name__ == "__main__":
    # Test the NiftiDataset
    image_folder = Path("data/images")  # Path to the input images
    mask_folder = Path("data/masks")   # Path to the segmentation masks

    # Create dataset instance
    dataset = NiftiDataset(image_folder, mask_folder, target_shape=(256, 256, 64))

    # Print dataset size
    print(f"Number of samples in dataset: {len(dataset)}")

    # いくつかのサンプルを確認
    for idx in range(min(3, len(dataset))):  # 最初の3つをチェック
        image, mask = dataset[idx]
        print(f"Sample {idx}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Unique mask values: {torch.unique(mask)}")
