import os
import nibabel as nib
import numpy as np
from PIL import Image


def convert_nifti_to_png(input_dir, output_dir, start_index=1, is_mask=False):
    """
    Convert NIfTI 3D images to PNG format for each axial slice with sequential numbering.

    Args:
        input_dir (str): Path to the directory containing NIfTI files (images or masks).
        output_dir (str): Path to the directory where PNG files will be saved.
        start_index (int): Starting index for numbering the output PNG files.
        is_mask (bool): Whether the input files are masks. If True, unique values will be mapped to 0, 127, 255.
    Returns:
        int: The next available index after processing all files in the input_dir.
    """
    # 入力ディレクトリ内のすべてのNIfTIファイルを取得
    nifti_files = sorted(os.listdir(input_dir))

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 現在のインデックス
    current_index = start_index

    # マッピング値
    mask_values = [0, 127, 255]

    # ファイルごとに処理
    for nifti_file in nifti_files:
        # ファイルパス
        input_path = os.path.join(input_dir, nifti_file)

        # NIfTIファイルを読み込む
        nifti_img = nib.load(input_path)
        data = nifti_img.get_fdata()

        # 各スライスをPNG形式で保存
        for slice_idx in range(data.shape[2]):  # Axialスライス
            slice_data = data[:, :, slice_idx]

            if is_mask:
                # マスクの場合、ユニークな値を0, 127, 255にマッピング
                unique_values = np.unique(slice_data)
                assert len(unique_values) <= len(
                    mask_values
                ), "More unique values in mask than expected."
                value_map = {v: mask_values[i] for i, v in enumerate(unique_values)}
                slice_data = np.vectorize(value_map.get)(slice_data).astype(np.uint8)
            else:
                # 正規化して0-255の範囲にスケーリング（画像データの場合）
                slice_data = (
                    (slice_data - slice_data.min())
                    / (slice_data.max() - slice_data.min())
                    * 255
                )
                slice_data = slice_data.astype(np.uint8)

            # 保存パス
            output_path = os.path.join(output_dir, f"{current_index:05d}.png")

            # 保存
            Image.fromarray(slice_data).save(output_path)
            current_index += 1

        print(f"Processed {nifti_file}: {data.shape[2]} slices saved.")

    return current_index


def main():
    # 元のデータディレクトリ
    base_dir = r"C:\Users\orilab\Desktop\masumoto\PL_2D\temp"
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")

    # 新しいデータディレクトリ
    new_base_dir = r"\Users\orilab\Desktop\masumoto\PL_2D\png_data0206"
    new_images_dir = os.path.join(new_base_dir, "images")
    new_masks_dir = os.path.join(new_base_dir, "masks")

    # ディレクトリ構造を作成し、変換を実行
    print("Processing images...")
    convert_nifti_to_png(
        images_dir, new_images_dir, start_index=1, is_mask=False
    )  # 画像は普通に変換

    print("Processing masks...")
    convert_nifti_to_png(
        masks_dir, new_masks_dir, start_index=1, is_mask=True
    )  # マスクは固定値にマッピング

    print("All files processed and saved to data2.")


if __name__ == "__main__":
    main()
