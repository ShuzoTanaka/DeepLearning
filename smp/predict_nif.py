import os
import numpy as np
import nibabel as nib
import cv2
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ====== 設定 ======
nifti_path = "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\DWI.nii.gz"  # 入力NIfTI
image_folder = "input_png"  # PNG保存先
output_folder = "output_png"  # 推論マスク保存先
output_nifti_path = "output_mask.nii.gz"

os.makedirs(image_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# ====== 1. NIfTIの読み込みとPNGスライス保存 ======
# NIfTI読み込み
nii = nib.load(nifti_path)
volume = nii.get_fdata()  # shape: (256, 256, 52, 12)
affine = nii.affine

# 最初のチャンネル（4次元目の index 0）を使用
for i in range(volume.shape[2]):
    slice_img = volume[:, :, i, 0]  # shape: (256, 256)
    norm_img = (
        (slice_img - slice_img.min()) / (np.ptp(slice_img) + 1e-8) * 255
    ).astype(np.uint8)
    print(norm_img.shape)
    cv2.imwrite(os.path.join(image_folder, f"slice_{i:03}.png"), norm_img)


# ====== 2. モデル読み込みと前処理 ======
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(
    "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\checkpoints\\20250623_2058_unet_resnet34.pth",
    map_location=DEVICE,
)
model.eval()

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocessing_fn(image)
    image = image.transpose(2, 0, 1).astype("float32")
    return torch.from_numpy(image).unsqueeze(0)


# ====== 3. PNG → 推論 → マスク保存 ======
png_files = sorted(os.listdir(image_folder))
for f in tqdm(png_files):
    path = os.path.join(image_folder, f)
    x_tensor = preprocess_image(path).to(DEVICE)
    with torch.no_grad():
        pred = model(x_tensor)
    mask = pred.squeeze().cpu().numpy()
    mask = np.argmax(mask, axis=0).astype(np.uint8)  # 0,1,2クラス
    cv2.imwrite(os.path.join(output_folder, f), mask * 127)  # 可視化用に127, 255など

# ====== 4. 推論マスクをNIfTI形式に再構成 ======
mask_slices = []
for f in sorted(os.listdir(output_folder)):
    m = cv2.imread(os.path.join(output_folder, f), cv2.IMREAD_GRAYSCALE)
    mask_slices.append(m // 127)  # 元のクラス0/1/2に戻す

mask_volume = np.stack(mask_slices, axis=-1).astype(np.uint8)
nifti_mask = nib.Nifti1Image(mask_volume, affine)
nib.save(nifti_mask, output_nifti_path)
