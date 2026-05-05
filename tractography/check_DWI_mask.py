import nibabel as nib
import numpy as np
import plotly.graph_objects as go

# NIfTIファイルを読み込み
mask_file = "0916DTINII_brain_mask.nii.gz"
mask_img = nib.load(mask_file)
mask_data = mask_img.get_fdata()

# マスクデータのボクセル値が0以上の領域を抽出
x, y, z = np.where(mask_data > 0)

# 3Dプロットを作成
fig = go.Figure(
    data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=2,
            color=z,  # z方向で色をつける
            colorscale="Viridis",  # 色スケール
            opacity=0.5,
        ),
    )
)

fig.update_layout(
    scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
    title="3D Visualization of Mask",
)

fig.show()
