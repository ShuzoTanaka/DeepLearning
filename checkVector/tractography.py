# 一応矢印が見れる。
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ファイルパスの設定
file_paths = {
    "V1": "yamadaYouko/Yamada_V1.nii.gz",
    "ROI": "yamadaYouko/Only2ROI.nii.gz",  # ROIファイル
}

# NIfTIデータの読み込み
data = {key: nib.load(path).get_fdata() for key, path in file_paths.items()}

# ROI中のボクセルを取得 (z=24とz=31のスライスでそれぞれ抽出)
z_slices = [24, 31]  # z=24 (Pythonのインデックスは0基準)
roi_indices = {z: np.argwhere((data["ROI"][:, :, z] > 0)) for z in z_slices}

# 3Dプロット
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# 各スライスについてベクトルを可視化
colors = {24: "blue", 31: "red"}
for z_slice, color in colors.items():
    for x, y in roi_indices[z_slice]:
        vector = data["V1"][x, y, z_slice, :]  # V1のベクトル (x, y, z方向の成分)
        ax.quiver(
            x,
            y,
            z_slice,  # 開始位置
            vector[0],
            vector[1],
            vector[2],  # ベクトル成分
            color=color,
            length=1,
            normalize=True,
            alpha=0.8,
        )

# 3Dプロットのフォーマット設定
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([0, data["ROI"].shape[0]])
ax.set_ylim([0, data["ROI"].shape[1]])
ax.set_zlim([20, 35])
ax.set_title("3D Visualization of Eigenvectors in ROI")

# 凡例を追加
for z_slice, color in colors.items():
    ax.plot([], [], [], color=color, label=f"z={z_slice}")
ax.legend(loc="upper left")

plt.show()
