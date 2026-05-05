import os
import nibabel as nib
import numpy as np

# ======== 設定 =========
input_dir = "masks"  # 入力フォルダ
output_dir = "output_nifti_change"  # 出力フォルダ
os.makedirs(output_dir, exist_ok=True)
# ======================

for filename in os.listdir(input_dir):
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        continue  # NIfTIファイル以外はスキップ

    filepath = os.path.join(input_dir, filename)

    # NIfTI読み込み
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.int32)  # 整数に変換
    unique_vals = np.unique(data)

    # 「0, 1, 2」すべてが含まれる場合だけ処理
    if set(unique_vals) == {0, 1, 2}:
        print(f"Processing {filename} ...")

        # 一時マスクを使って 1 ↔ 2 を入れ替え
        mask1 = data == 1
        mask2 = data == 2
        data[mask1] = 2
        data[mask2] = 1

        # 保存
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, os.path.join(output_dir, filename))
    else:
        print(f"Skipping {filename} (labels found: {unique_vals})")
