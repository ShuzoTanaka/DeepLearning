import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.direction import peaks_from_model
from dipy.reconst.dti import fractional_anisotropy
from fury import window, actor

# 拡散データとマスクを読み込む
dwi_img = nib.load("iida_dcm2niix.nii.gz")
dwi_data = dwi_img.get_fdata()

mask_img = nib.load("NifTI_brain_mask.nii.gz")
mask_data = mask_img.get_fdata()

# bvals, bvecsを読み込む
bvals, bvecs = read_bvals_bvecs("iida_dcm2niix.bval", "iida_dcm2niix.bvec")

# 勾配テーブルを作成
gtab = gradient_table(bvals, bvecs)

# ROIデータを読み込む
roi_img = nib.load("ROI.nii.gz")
roi_data = roi_img.get_fdata()

# ROIデータからシードポイントを生成（値が1のボクセルを使用）
seeds = utils.seeds_from_mask(roi_data > 0, affine=dwi_img.affine, density=1)
