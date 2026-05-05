import os
import numpy as np
from PIL import Image


def compute_class_weights_from_png_folder(folder_path, num_classes=None, eps=1e-6):
    """
    folder_path 内の .png ラベル画像を読み込み、
    画素値（クラス）ごとのピクセル総数・頻度・重み (1/freq) を計算する。

    想定：
        - 画素値 = クラスID（0,1,2,...）
        - グレースケール or 1ch のラベル画像 1.01435120e-06  6.27341558e-04  1.18296227e-03
    """
    total_counts = None

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue

        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path)

        # 必ず1chラベルとして扱う
        img = img.convert("L")
        arr = np.array(img, dtype=np.int64)  # 画素値 = クラスID
        flat = arr.flatten()

        if num_classes is None:
            max_label = flat.max()
            if total_counts is None:
                total_counts = np.zeros(max_label + 1, dtype=np.int64)
            elif max_label + 1 > len(total_counts):
                # 新しいクラス値が出てきたら配列を拡張
                total_counts = np.pad(
                    total_counts, (0, max_label + 1 - len(total_counts))
                )
        else:
            if total_counts is None:
                total_counts = np.zeros(num_classes, dtype=np.int64)

        # この画像のクラスごとのカウント
        counts = np.bincount(flat, minlength=len(total_counts))
        total_counts += counts

    if total_counts is None:
        raise ValueError("指定フォルダに PNG 画像がありません。")

    total_pixels = total_counts.sum()
    freq = total_counts.astype(np.float64) / total_pixels  # クラス頻度

    # 1 / freq で重みを計算（ゼロ除算回避のため eps 加算）
    raw_weights = 1.0 / (freq + eps)

    # スケールしやすいように、平均が1になるよう正規化（任意）
    weights = raw_weights / raw_weights.mean()

    return total_counts, freq, weights


if __name__ == "__main__":
    folder = "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\data_split\\train\\temp"  # ← フォルダパスをここに
    counts, freq, weights = compute_class_weights_from_png_folder(folder)

    print("クラスごとのピクセル数:", counts)
    print("クラスごとの頻度:", freq)
    print("クラスごとの重み (mean=1 に正規化済):", weights)

    # PyTorch で使いたい場合の例
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print("PyTorch 用 tensor:", class_weights_tensor)
