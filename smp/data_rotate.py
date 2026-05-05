import os
from PIL import Image

# 対象フォルダのパスを指定（例: "./images"）
folder_path = "C:\\Users\\orilab\\Desktop\\masumoto\\smp\\data_split_temp\\test\\mask"


# フォルダ内のファイルを走査
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".png"):
        file_path = os.path.join(folder_path, filename)

        # 画像を開いて左に90度回転（反時計回り）
        with Image.open(file_path) as img:
            rotated_img = img.rotate(90, expand=True)
            rotated_img.save(file_path)

print("すべてのPNG画像を90度左回転させました。")
