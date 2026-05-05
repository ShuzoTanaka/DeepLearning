import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset
import cv2
import albumentations as albu
import segmentation_models_pytorch as smp

# 保存先フォルダ（なければ作成）
output_dir = "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\prediction_output"
os.makedirs(output_dir, exist_ok=True)

test_dir = "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\data_split_temp_copy\\test"


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


ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

test_dataset = Dataset(
    os.path.join(test_dir, "image"),
    os.path.join(test_dir, "mask"),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=["background", "nerve", "spinal"],
)

DEVICE = "cuda"
best_model = torch.load(
    f"C:\\Users\\orilab\\Desktop\\masumoto\\smp\\checkpoints\\20250623_2058_unet_resnet34.pth"
)

# 推論処理（バッチ単位）
for i in tqdm(range(len(test_dataset))):
    image_path = test_dataset.images_fps[i]
    image_name = (
        os.path.basename(image_path).replace(".jpg", ".png").replace(".jpeg", ".png")
    )

    # 入力画像の取得と予測
    image, _ = test_dataset[i]
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        pr_mask = best_model.predict(x_tensor)

    # 後処理
    pr_mask = pr_mask.squeeze().cpu().numpy().round()  # shape: (3, H, W)
    pr_mask = pr_mask.transpose(1, 2, 0)  # shape: (H, W, 3)

    # 各クラスに対応するラベル値（例: 0, 127, 255）
    class_values = [0, 127, 255]
    mask_label = np.zeros(pr_mask.shape[:2], dtype=np.uint8)  # shape: (H, W)

    for class_index, label_value in enumerate(class_values):
        mask_label[pr_mask[..., class_index] == 1] = label_value

    # 保存（PNG形式で可視化しやすい）
    save_path = os.path.join(output_dir, image_name)
    cv2.imwrite(save_path, mask_label)

print("✅ 全ての予測マスクを保存しました。")
