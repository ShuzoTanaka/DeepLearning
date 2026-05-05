# ライブラリ読み込み
import glob
import cv2
import numpy as np
import os
import os.path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils

import albumentations as albu
import matplotlib.pyplot as plt
from PIL import Image


# テンソル化
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# 前処理
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# データセット
class Dataset(BaseDataset):
    # CLASSES = ['background', 'SC']
    CLASSES = ["background", "nerve", "spinal"]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [0, 127, 255]
        # self.class_values = [classes.index(cls) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        masks = np.array([(mask == v) for v in self.class_values])
        mask = np.stack(masks, axis=-1).astype("float")

        # test_mask_name.append(self.masks_fps[i])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


test_dir = "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\data_split\\test"
# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    os.path.join(test_dir, "image"),
    os.path.join(test_dir, "mask"),
    classes=["background", "nerve", "spinal"],
)
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
# create test dataset
test_dataset = Dataset(
    os.path.join(test_dir, "image"),
    os.path.join(test_dir, "mask"),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=["background", "nerve", "spinal"],
)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

# 1つだけ可視化
n = 55
image_vis = test_dataset_vis[n][0].astype("uint8")
image, gt_mask = test_dataset[n]

gt_mask = gt_mask.squeeze()
gt_mask = gt_mask.transpose(1, 2, 0)

DEVICE = "cuda"
best_model = torch.load(
    f"C:\\Users\\orilab\\Desktop\\masumoto\\smp\\checkpoints\\20250623_2058_unet_resnet34.pth"
)
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
pr_mask = best_model.predict(x_tensor)
pr_mask = pr_mask.squeeze().cpu().numpy().round()
pr_mask = pr_mask.transpose(1, 2, 0)
