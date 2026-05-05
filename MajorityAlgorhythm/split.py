import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 設定 ===
nifti_dir = "C:/Users/orilab/Desktop/masumoto/MajorityAlgorhythm/niftidata/masks"  # NIfTIファイルのフォルダ
output_base_dir = "pngData/masks"  # 出力フォルダ
scale_factor = 3 / 1.25  # = 2.4


# 画像保存関数（リサイズ対応）
def save_slices(image_array, save_dir, orientation, rescale_axis=None):
    """MRI画像をスライスごとにPNGとして保存（必要ならリサイズ）"""
    os.makedirs(save_dir, exist_ok=True)

    num_slices = image_array.shape[2]  # スライス数
    for i in range(num_slices):
        slice_img = image_array[:, :, i]

        # Coronal・Sagittal の場合、スケール調整
        if rescale_axis == "height":
            new_height = int(slice_img.shape[0] * scale_factor)  # 縦方向に拡大
            slice_img = cv2.resize(
                slice_img,
                (slice_img.shape[1], new_height),
                interpolation=cv2.INTER_LINEAR,
            )
        elif rescale_axis == "width":
            new_width = int(slice_img.shape[1] * scale_factor)  # 横方向に拡大（修正）
            slice_img = cv2.resize(
                slice_img,
                (new_width, slice_img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # 画像の保存パス
        save_path = os.path.join(save_dir, f"{orientation}_{i:03d}.png")

        # 画像として保存（グレースケール）
        plt.imsave(
            save_path,
            slice_img,
            cmap="gray",
            vmin=np.min(image_array),
            vmax=np.max(image_array),
        )


# === NIfTIファイルを処理 ===
for nifti_file in tqdm(os.listdir(nifti_dir)):
    if nifti_file.endswith(".nii") or nifti_file.endswith(".nii.gz"):
        case_name = os.path.splitext(nifti_file)[0].replace(".nii", "")

        case_dir = os.path.join(output_base_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        axial_dir = os.path.join(case_dir, "Axial")
        coronal_dir = os.path.join(case_dir, "Coronal")
        sagittal_dir = os.path.join(case_dir, "Sagittal")
        os.makedirs(axial_dir, exist_ok=True)
        os.makedirs(coronal_dir, exist_ok=True)
        os.makedirs(sagittal_dir, exist_ok=True)

        # NIfTIファイルをロード
        nifti_path = os.path.join(nifti_dir, nifti_file)
        nifti_img = nib.load(nifti_path)
        img_data = nifti_img.get_fdata()  # numpy配列に変換

        # 軸のスライス方法を明示的に指定
        axial_slices = img_data[:, :, :]  # Axial（元のデータ: Z軸でスライス）
        sagittal_slices = np.transpose(img_data, (2, 1, 0))  # Sagittal（X軸でスライス）
        coronal_slices = np.transpose(img_data, (0, 2, 1))  # Coronal（Y軸でスライス）

        # 各軸のスライスを保存
        save_slices(axial_slices, axial_dir, "Axial")  # Axial: そのまま
        save_slices(
            sagittal_slices, sagittal_dir, "Sagittal", rescale_axis="height"
        )  # Sagittal: 縦を 2.4 倍に拡大
        save_slices(
            coronal_slices, coronal_dir, "Coronal", rescale_axis="width"
        )  # Coronal: 横を 2.4 倍に拡大（修正）

        print(f"Processed: {nifti_file}")

print("すべてのNIfTIファイルの変換が完了しました。")
