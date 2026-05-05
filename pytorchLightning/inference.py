# 二つのニフティファイル間のDice係数を表示するスクリプト　08/14
import nibabel as nib
import numpy as np

# ===== ユーザー設定 =====
pred_path = "C:/Users/orilab/Desktop/masumoto/pytorchLightning/nifti_predictions-2025-09-19_11-04-06/sample_7_pred.nii.gz"
gt_path = "C:/Users/orilab/Desktop/masumoto/pytorchLightning/nifti_predictions-2025-09-19_11-04-06/sample_7_gt.nii.gz"  # 正解マスク
num_classes = 3  # 背景含めたクラス数
# ========================

# ===== NIfTI読み込み =====
pred_img = nib.load(pred_path)
gt_img = nib.load(gt_path)

pred_mask = pred_img.get_fdata().astype(np.int32)
gt_mask = gt_img.get_fdata().astype(np.int32)


# ===== Dice係数計算関数 =====
def dice_coefficient(mask1, mask2):
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    size1 = np.sum(mask1 > 0)
    size2 = np.sum(mask2 > 0)
    if size1 + size2 == 0:
        return 1.0  # 両方空なら完全一致
    return 2.0 * intersection / (size1 + size2)


# ===== クラスごとのDice計算 =====
for c in range(num_classes):
    pred_bin = pred_mask == c
    gt_bin = gt_mask == c
    dice = dice_coefficient(pred_bin, gt_bin)
    print(f"Class {c} Dice: {dice:.4f}")
