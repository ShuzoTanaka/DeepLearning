import os
import nibabel as nib


def nii_to_niigz(nii_path: str, out_path: str | None = None):
    """
    .nii → .nii.gz に正しく変換する
    """
    if not nii_path.endswith(".nii"):
        raise ValueError("Input file must be .nii")

    if out_path is None:
        out_path = nii_path + ".gz"  # xxx.nii.gz

    # NIfTI を読み込み
    nii = nib.load(nii_path)

    # .nii.gz として保存（gzip 圧縮される）
    nib.save(nii, out_path)

    print(f"Converted: {nii_path} -> {out_path}")


# ===== 使用例 =====
nii_to_niigz("Dataset\\imagesTs\\4.nii")
nii_to_niigz("Dataset\\imagesTs\\5.nii")
nii_to_niigz("Dataset\\imagesTs\\7.nii")
