import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 固有ベクトルや固有値のNIfTIファイルパス
file_path_V1 = "yamadaYouko\Yamada_V1.nii.gz"  # 1st eigenvector file
file_path_L1 = "yamadaYouko\Yamada_L1.nii.gz"  # 1st eigenvalue file

# NIfTIファイルの読み込み
nii_V1 = nib.load(file_path_V1)
nii_L1 = nib.load(file_path_L1)

# データを取得
data_V1 = nii_V1.get_fdata()  # V1: shape = (x, y, z, 3)
data_L1 = nii_L1.get_fdata()  # L1: shape = (x, y, z)

# データの確認
print("V1 Shape:", data_V1.shape)  # Example: (128, 128, 70, 3)
print("L1 Shape:", data_L1.shape)  # Example: (128, 128, 70)

# 特定のボクセルでの固有ベクトルと固有値を確認 (i, j, k) = (64, 64, 35)
i, j, k = 64, 64, 35
V1_voxel = data_V1[i, j, k, :]  # [x, y, z]
L1_voxel = data_L1[i, j, k]  # Scalar

print(f"Voxel ({i}, {j}, {k}) V1 (eigenvector): {V1_voxel}")
print(f"Voxel ({i}, {j}, {k}) L1 (eigenvalue): {L1_voxel}")

# 任意のスライスで可視化 (例: z=35)
Y_slice = 111

# 固有ベクトルの方向を可視化 (矢印プロット)
plt.figure(figsize=(10, 10))
x, y = np.meshgrid(range(data_V1.shape[0]), range(data_V1.shape[1]))
u = data_V1[:, Y_slice, :, 0]  # x方向の成分
v = data_V1[:, Y_slice, :, 1]  # y方向の成分

plt.quiver(x, y, u, v, scale=10, color="blue", alpha=0.5)
plt.title(f"Eigenvectors (V1) at Y={Y_slice}")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# 固有値のスライス表示
plt.figure(figsize=(8, 8))
plt.imshow(data_L1[:, Y_slice, :], cmap="hot", interpolation="nearest")
plt.colorbar(label="Eigenvalue (L1)")
plt.title(f"Eigenvalue (L1) at y={Y_slice}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
