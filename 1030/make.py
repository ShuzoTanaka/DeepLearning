import nibabel as nib
import numpy as np

# NIfTI 読み込み
nii = nib.load(r"C:\Users\orilab\Desktop\masumoto\1030\pred_out_test\case040.nii.gz")
data = nii.get_fdata()  # float64 になる点に注意

# [x, y, z] = [0, 0, 0] のボクセルを 2 にする
data[10, 10, 10] = 2

# 元のヘッダ・アフィンを保持して保存
new_nii = nib.Nifti1Image(data, affine=nii.affine, header=nii.header)
nib.save(new_nii, "output.nii.gz")
