from pathlib import Path
import shutil

# ===== 設定 =====
SRC_ROOT = Path(r"C:\Users\orilab\Desktop\nifti_out")
DST_ROOT = Path(r"C:\Users\orilab\Desktop\nifti_folder")

DST_ROOT.mkdir(parents=True, exist_ok=True)


def is_nifti(p: Path) -> bool:
    return p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))


def copy_with_rename(src: Path, dst_dir: Path):
    """
    同名ファイルがあれば _001, _002 ... を付けてコピー
    """
    name = src.name

    # 拡張子を正しく扱う
    if name.endswith(".nii.gz"):
        stem = name[:-7]
        ext = ".nii.gz"
    else:
        stem = name[:-4]
        ext = ".nii"

    dst = dst_dir / name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i:03d}{ext}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1


count = 0
for nifti in SRC_ROOT.rglob("*"):
    if not is_nifti(nifti):
        continue

    out_path = copy_with_rename(nifti, DST_ROOT)
    print(f"copied: {nifti} -> {out_path}")
    count += 1

print(f"\nDone. total nifti files copied: {count}")
