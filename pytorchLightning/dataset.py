# dataset.py
from pathlib import Path
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ───────────────────────────────────────────
# 3D NIfTI Dataset with intensity→class mapping
# ───────────────────────────────────────────
class NiftiDataset(Dataset):
    """
    3D セグメンテーション用 Dataset
    - 画像: NIfTI (.nii / .nii.gz)
    - マスク: NIfTI (.nii / .nii.gz)
    - 返り値: image, mask（ともに [1, H, W, D]）
    - mask はクラスID（0/1/2 …）で返す

    intensity→class の既定対応:
        LABEL_VALUES = [0, 127, 255]  # 背景/神経/硬膜管など
        → 0→class0, 127→class1, 255→class2

    ただし、ファイルのユニーク値が {0,1} や {0,1,2} の場合も自動で安全にマッピングします。
    """

    # 参考用クラス名（任意）
    CLASSES = ["background", "nerve", "spinal"]
    # 既定の輝度値（順序＝クラス順）
    LABEL_VALUES = [0, 1, 2]

    def __init__(
        self,
        image_folder,
        mask_folder,
        target_shape=(256, 256, 64),
        case_list=None,
        transform=None,
        label_values=None,  # 例: [0,127,255] など。未指定なら既定を使用
    ):
        """
        Args:
            image_folder (str|Path): 画像NIfTIのディレクトリ
            mask_folder  (str|Path): マスクNIfTIのディレクトリ
            target_shape (H, W, D): リサイズ後の形状
            case_list (list|None): 対象症例ID（拡張子除くファイル名）でフィルタ
            transform: MONAIのdict変換（{'image':..., 'mask':...}）
            label_values (list|None): 輝度値のリスト（順序＝クラス順）
        """
        image_folder = Path(image_folder)
        mask_folder = Path(mask_folder)

        # .nii / .nii.gz の両方を拾う
        all_image_paths = sorted(
            list(image_folder.glob("*.nii")) + list(image_folder.glob("*.nii.gz"))
        )
        all_mask_paths = sorted(
            list(mask_folder.glob("*.nii")) + list(mask_folder.glob("*.nii.gz"))
        )

        if case_list is not None:
            case_set = set(case_list)

            def stem(p: Path):  # 複数拡張子にも対応
                return p.name.split(".")[0]

            filtered_images = [p for p in all_image_paths if stem(p) in case_set]
            filtered_masks = [p for p in all_mask_paths if stem(p) in case_set]
            assert len(filtered_images) == len(filtered_masks), (
                f"Mismatch: {len(filtered_images)} images vs {len(filtered_masks)} masks\n"
                f"Missing in images: {case_set - {stem(p) for p in filtered_images}}\n"
                f"Missing in masks:  {case_set - {stem(p) for p in filtered_masks}}"
            )
            self.image_paths = sorted(filtered_images, key=lambda p: stem(p))
            self.mask_paths = sorted(filtered_masks, key=lambda p: stem(p))
        else:
            # ファイル名（拡張子除く）が一致するペアのみ採用
            def stem(p: Path):
                return p.name.split(".")[0]

            images_by_stem = {stem(p): p for p in all_image_paths}
            masks_by_stem = {stem(p): p for p in all_mask_paths}
            common = sorted(set(images_by_stem) & set(masks_by_stem))
            self.image_paths = [images_by_stem[s] for s in common]
            self.mask_paths = [masks_by_stem[s] for s in common]

        # 最終チェック
        for ip, mp in zip(self.image_paths, self.mask_paths):
            assert (
                ip.name.split(".")[0] == mp.name.split(".")[0]
            ), f"Unmatched pair: {ip} vs {mp}"

        self.target_shape = tuple(target_shape)
        self.transform = transform
        self.label_values = (
            list(label_values) if label_values is not None else list(self.LABEL_VALUES)
        )

        # 事前に intensity→class の基本LUT（辞書）を構築
        self._label_to_class = {int(v): idx for idx, v in enumerate(self.label_values)}

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max - arr_min < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)

    def _intensity_to_class(self, mask_np: np.ndarray) -> np.ndarray:
        """
        輝度値→クラスID(0/1/2 …)に変換して [D,H,W] または [H,W,D] の形を保つ。
        - 既定: self.label_values のLUTでマップ(例: 0→0, 127→1, 255→2)
        - ただし、ファイル内ユニーク値が {0,1} や {0,1,2} の場合は自動で安全に対応
          * {0,1} → 0→0, 1→1
          * {0,1,2} → 0→0, 1→1, 2→2
        それ以外は self._label_to_class に存在しない値は 0(背景)へフォールバック。
        """
        uniq = np.unique(mask_np).astype(int)
        uniq_set = set(uniq.tolist())

        # 安全な自動対応
        if uniq_set == {0, 1}:
            lut = {0: 0, 1: 1}
        elif uniq_set == {0, 1, 2}:
            lut = {0: 0, 1: 1, 2: 2}
        else:
            # 既定（例: 0/127/255）に従う。未登録値は 0 に落とす。
            lut = self._label_to_class

        mapped = np.vectorize(lambda v: int(lut.get(int(v), 0)))(mask_np).astype(
            np.int64
        )
        return mapped

    def __getitem__(self, idx):
        """
        Returns:
            image: torch.FloatTensor [1, H, W, D]
            mask:  torch.LongTensor  [1, H, W, D]  (中身はクラスID)
        """
        # ── 1) load ─────────────────────────────
        image_np = nib.load(self.image_paths[idx]).get_fdata()
        mask_np = nib.load(self.mask_paths[idx]).get_fdata()

        # 次元は [H,W,D] を想定（NIfTIにより [X,Y,Z]）
        # 必要ならここで転置を追加してください。

        # ── 2) normalize image ──────────────────
        image_np = self._normalize01(image_np)  # [H,W,D] float32

        # ── 3) intensity → class-id ────────────
        # マスクは整数想定（0,1 / 0,1,2 / 0,127,255 など）
        mask_idx_np = self._intensity_to_class(mask_np)  # [H,W,D] int64

        # ── 4) to torch [C,H,W,D] ──────────────
        image = torch.from_numpy(image_np).float().unsqueeze(0)  # [1,H,W,D]
        mask = torch.from_numpy(mask_idx_np).long().unsqueeze(0)  # [1,H,W,D]

        # ── 5) resize to target_shape ──────────
        # image: trilinear / mask: nearest
        # 入力は [N,C,H,W,D] なので一旦 N 次元を付与
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # → [1,H,W,D]
        mask = (
            F.interpolate(
                mask.unsqueeze(0).float(), size=self.target_shape, mode="nearest"
            )
            .squeeze(0)
            .long()
        )  # → [1,H,W,D]

        # ── 6) MONAI transforms (dict) ─────────
        if self.transform:
            sample = {"image": image, "mask": mask}
            sample = self.transform(sample)
            image, mask = sample["image"].float(), sample["mask"].long()

        return image, mask


# 簡単な動作確認
if __name__ == "__main__":
    # 例:
    ds = NiftiDataset(
        image_folder="data/images",
        mask_folder="data/masks",
        target_shape=(256, 256, 64),
        # label_values=[0, 127, 255],  # 必要なら明示
        transform=None,  # MONAIのCompose({...})を渡せます
    )
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        img, msk = ds[0]
        print("Image:", img.shape, img.dtype)  # [1,H,W,D], float32
        print("Mask :", msk.shape, msk.dtype, torch.unique(msk))  # [1,H,W,D], long
