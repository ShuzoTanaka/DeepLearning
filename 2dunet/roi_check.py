#!/usr/bin/env python
import os
import argparse

import numpy as np
import nibabel as nib


def get_label_bbox(data: np.ndarray, label_value: int = 1):
    """
    data: 3D array (X, Y, Z) of labels
    label_value: ROI のラベル値 (デフォルト 1)

    戻り値: (x_min, x_max, y_min, y_max, z_min, z_max) または None（ラベルなし）
    """
    mask = data == label_value

    if not np.any(mask):
        return None

    xs, ys, zs = np.where(mask)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()

    return x_min, x_max, y_min, y_max, z_min, z_max


def main():
    parser = argparse.ArgumentParser(
        description="マスクNIfTI中の label=1 の X/Y/Z 範囲を一覧表示するスクリプト"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="マスクNIfTI（.nii / .nii.gz）が入っているフォルダ",
    )
    parser.add_argument(
        "--label_value",
        type=int,
        default=1,
        help="ROI として扱うラベル値（デフォルト: 1）",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="結果を保存するCSVファイルパス（指定しない場合は標準出力のみ）",
    )
    args = parser.parse_args()

    mask_dir = args.mask_dir
    label_value = args.label_value

    files = [
        f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    files.sort()

    if not files:
        print("マスクファイル（.nii / .nii.gz）が見つかりませんでした。")
        return

    print(f"フォルダ: {mask_dir}")
    print(f"対象ラベル値: {label_value}")
    print("-" * 80)

    results = []
    header = [
        "filename",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "has_label",
    ]
    print("\t".join(header))

    for fname in files:
        path = os.path.join(mask_dir, fname)
        try:
            nii = nib.load(path)
            data = nii.get_fdata().astype(np.int32)  # ラベル画像なので int に変換
        except Exception as e:
            print(f"{fname}\t読み込みエラー: {e}")
            continue

        bbox = get_label_bbox(data, label_value=label_value)

        if bbox is None:
            row = [fname, "", "", "", "", "", "", "0"]
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
            row = [
                fname,
                str(x_min),
                str(x_max),
                str(y_min),
                str(y_max),
                str(z_min),
                str(z_max),
                "1",
            ]

        print("\t".join(row))
        results.append(row)

    # CSV 保存（必要なら）
    if args.save_csv is not None:
        import csv

        csv_path = args.save_csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
        print(f"\n結果をCSVに保存しました: {csv_path}")


if __name__ == "__main__":
    main()
