# 2D U-Net で予測 → NIfTI保存（3D/4D入力対応）→ (256, 256, 64) にリサンプリング
# - マルチクラスNIfTI（0/1/2）も保存
# - ラベル1のバイナリNIfTIも保存

import os
import numpy as np
import nibabel as nib
import cv2
import torch
import segmentation_models_pytorch as smp

from tqdm import tqdm
from scipy.ndimage import zoom

# ====== パス設定（必要に応じて変更） ======
directory_path = (
    r"C:\Users\orilab\Desktop\masumoto\pytorchLightning\pred_2D_0814\sample7"
)
nifti_path = r"C:\Users\orilab\Desktop\masumoto\pytorchLightning\nifti_predictions-2025-08-14_18-17-02\sample_7_image.nii.gz"
model_path = (
    r"C:\Users\orilab\Desktop\masumoto\pytorchLightning\20250623_2058_unet_resnet34.pth"
)

image_folder = os.path.join(directory_path, "image_folder")
output_folder = os.path.join(directory_path, "output_folder")
os.makedirs(image_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 出力ファイル名
output_multiclass_path = os.path.join(directory_path, "multiclass_resampled.nii.gz")
output_binary_path = os.path.join(directory_path, "label1_resampled.nii.gz")

# リサンプリング後の目標形状
TARGET_SHAPE = (256, 256, 64)  # (H, W, S)

# ====== 1. NIfTIの読み込みとPNGスライス保存 ======
if not os.path.isfile(nifti_path):
    raise FileNotFoundError(f"NIfTIファイルが見つかりません: {nifti_path}")

nii = nib.load(nifti_path)
volume = nii.get_fdata()  # 3D: (H,W,S) or 4D: (H,W,S,C)
affine = nii.affine
is_4d = volume.ndim == 4
channel_index = 0  # 4Dのとき使用するチャンネル

num_slices = volume.shape[2]
for i in range(num_slices):
    if is_4d:
        slice_img = volume[:, :, i, channel_index]
    else:
        slice_img = volume[:, :, i]

    # 0-255へ正規化（一定値画像ケア）
    vmin, vmax = slice_img.min(), slice_img.max()
    if vmax > vmin:
        norm_img = ((slice_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        norm_img = np.zeros_like(slice_img, dtype=np.uint8)

    cv2.imwrite(os.path.join(image_folder, f"slice_{i:03}.png"), norm_img)

# ====== 2. モデル読み込みと前処理 ======
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 学習時に torch.save(model) した前提
model = torch.load(model_path, map_location=DEVICE)
model.eval()

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def preprocess_image(path):
    image = cv2.imread(path)  # BGR
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocessing_fn(image)  # encoderに合わせた正規化
    image = image.transpose(2, 0, 1).astype("float32")
    return torch.from_numpy(image).unsqueeze(0)  # (1,C,H,W)


# ====== 3. PNG → 推論（in-memoryでマスク蓄積）＋ 可視化PNG保存 ======
png_files = sorted(os.listdir(image_folder))
mask_slices = []  # ここに各スライスの 0/1/2 ラベルを積む

with torch.no_grad():
    for f in tqdm(png_files, desc="Infer"):
        path = os.path.join(image_folder, f)
        x_tensor = preprocess_image(path).to(DEVICE)
        pred = model(x_tensor)  # (1, C, H, W)
        logits = pred.squeeze(0).cpu().numpy()  # (C, H, W)
        mask = np.argmax(logits, axis=0).astype(np.uint8)  # (H, W): 0/1/2

        # in-memoryへ
        mask_slices.append(mask)

        # 可視化PNG（0/1/2→0/127/254）
        vis = (mask * 127).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f), vis)

# ====== 4. マスクを3Dボリューム化 ======
mask_volume = np.stack(mask_slices, axis=-1).astype(np.uint8)  # (H, W, S)


# ====== 5. (255, 255, 64) へリサンプリング（最近傍）＋ アフィン更新 ======
def resample_labels_to_shape(label_vol, affine, target_shape):
    """
    label_vol: (H, W, S) の整数ラベル配列（0/1/2）
    affine:    4x4 アフィン
    target_shape: (Ht, Wt, St)

    返り値:
      resampled_vol:  (Ht, Wt, St)  最近傍補間（order=0）
      new_affine:     FOVを維持するためスケールを調整したアフィン
    """
    old_shape = np.array(label_vol.shape, dtype=float)  # (H, W, S)
    new_shape = np.array(target_shape, dtype=float)

    # ndimage.zoom の倍率
    zoom_factors = new_shape / old_shape

    # 最近傍補間（ラベルが混ざらない）
    resampled = zoom(label_vol, zoom=zoom_factors, order=0, prefilter=False)

    # FOVを維持：各軸のvoxelベクトル（affineの前3列）を old_dim/new_dim 倍
    scale_cols = old_shape / new_shape
    new_affine = affine.copy()
    new_affine[:3, 0] *= scale_cols[0]  # i軸
    new_affine[:3, 1] *= scale_cols[1]  # j軸
    new_affine[:3, 2] *= scale_cols[2]  # k軸

    return resampled.astype(label_vol.dtype), new_affine


mask_resampled, affine_resampled = resample_labels_to_shape(
    mask_volume, affine, TARGET_SHAPE
)

# ====== 6. NIfTI作成・保存 ======
# 6-1) マルチクラス（0/1/2）
nifti_multiclass = nib.Nifti1Image(mask_resampled, affine_resampled, header=nii.header)
nib.save(nifti_multiclass, output_multiclass_path)

# 6-2) ラベル1のみ抽出（バイナリ）
binary_data = (mask_resampled == 1).astype(np.uint8)
nifti_binary = nib.Nifti1Image(binary_data, affine_resampled, header=nii.header)
nib.save(nifti_binary, output_binary_path)

print(
    "Saved multiclass:",
    output_multiclass_path,
    "shape:",
    mask_resampled.shape,
    "dtype:",
    mask_resampled.dtype,
)
print(
    "Saved label-1 bin:",
    output_binary_path,
    "shape:",
    binary_data.shape,
    "dtype:",
    binary_data.dtype,
)
