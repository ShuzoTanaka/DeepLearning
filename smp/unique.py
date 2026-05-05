from PIL import Image
import numpy as np


def list_unique_pixel_values(png_path):
    # 画像を読み込み（Lにすると1chのグレースケールラベルとして扱いやすい）
    img = Image.open(png_path).convert("L")
    arr = np.array(img)

    # ユニークな画素値を取得
    unique_vals = np.unique(arr)

    print("画像パス:", png_path)
    print("画像サイズ:", arr.shape)
    print("含まれる画素値（ユニーク）:", unique_vals.tolist())

    return unique_vals


# 使用例
png_path = (
    "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\data_split\\train\\mask\\00923.png"
)
unique_values = list_unique_pixel_values(png_path)
