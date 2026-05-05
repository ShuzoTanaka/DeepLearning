from pathlib import Path
import subprocess
import pydicom
import nibabel as nib
import numpy as np

# ===== 設定 =====
IN_ROOT = Path(r"C:\Users\orilab\Desktop\data_list")
OUT_ROOT = Path(r"C:\Users\orilab\Desktop\nifti_out")

# ★ここだけあなたのパスに固定
DCM2NIIX_EXE = r"C:\Users\orilab\Desktop\MRIcroGL\Resources\dcm2niix.exe"

EXTRACT_FIRST_VOL = True  # 4Dなら[...,0]だけ保存して3D化
GZ = True  # nii.gz

OUT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def try_read_tags(any_file: Path):
    try:
        ds = pydicom.dcmread(
            str(any_file),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                (0x0010, 0x0020),  # PatientID
                (0x0020, 0x0011),  # SeriesNumber
                (0x0020, 0x000E),  # SeriesInstanceUID
            ],
        )
    except Exception:
        return None

    patient_id = getattr(ds, "PatientID", None)
    series_no = getattr(ds, "SeriesNumber", None)
    series_uid = getattr(ds, "SeriesInstanceUID", None)

    if patient_id or series_uid:
        return (patient_id or "Unknown", series_no, series_uid or "NoUID")
    return None


def find_leaf_series_dirs(root: Path):
    """
    DICOMを含む末端（leaf）フォルダだけ返す
    """
    dicom_dirs = []

    # 「直下に読めるDICOMがある」ディレクトリを集める（高速寄り）
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        ok = False
        for f in d.iterdir():
            if f.is_file() and try_read_tags(f) is not None:
                ok = True
                break
        if ok:
            dicom_dirs.append(d)

    dicom_set = set(dicom_dirs)

    leafs = []
    for d in dicom_dirs:
        has_child = False
        for other in dicom_set:
            if other == d:
                continue
            if other.is_relative_to(d):
                has_child = True
                break
        if not has_child:
            leafs.append(d)

    leafs.sort(key=lambda x: len(x.parts), reverse=True)
    return leafs


def extract_first_volume_inplace(nifti_path: Path):
    img = nib.load(str(nifti_path))

    # 4Dじゃない / 4Dでも1ボリュームしかないなら何もしない
    shape = img.shape
    if len(shape) != 4 or shape[3] <= 1:
        return

    # 変なdtype（RGBなど = kind 'V'）はスキップ
    dt = img.get_data_dtype()
    if getattr(dt, "kind", None) == "V":
        print(f"[SKIP] non-numeric dtype (kind=V): {nifti_path.name}")
        return

    # 数値型だけ、[...,0] を抜いて保存
    data0 = np.asanyarray(img.dataobj)[..., 0].astype(np.float32, copy=False)
    new_img = nib.Nifti1Image(data0, img.affine, img.header)
    nib.save(new_img, str(nifti_path))


leaf_series_dirs = find_leaf_series_dirs(IN_ROOT)
print(f"Found leaf series dirs: {len(leaf_series_dirs)}")
print(f"Using dcm2niix: {DCM2NIIX_EXE}")

for series_dir in leaf_series_dirs:
    # leaf内のどれか1ファイルでタグ取得
    tags = None
    picked = None
    for f in series_dir.rglob("*"):
        if f.is_file():
            tags = try_read_tags(f)
            if tags is not None:
                picked = f
                break

    if tags is None:
        print(f"[SKIP] cannot read tags: {series_dir}")
        continue

    patient_id, series_no, series_uid = tags

    if series_no is not None:
        base = f"{patient_id}_S{int(series_no):03d}"
    else:
        base = f"{patient_id}_{series_uid}"

    safe_pid = safe_filename(patient_id)
    safe_base = safe_filename(base)

    out_dir = OUT_ROOT / safe_pid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"convert: {series_dir} -> {safe_base}  (picked={picked.name})")

    # 既に変換済みならスキップ（dcm2niixが a/b を付けるのを防ぐ）
    already = list(out_dir.glob(f"{safe_base}*.nii*"))
    if already:
        print(f"[SKIP] already converted: {series_dir} -> {already[0].name}")
        continue

    result = subprocess.run(
        [
            DCM2NIIX_EXE,
            "-z",
            "y" if GZ else "n",
            "-b",
            "n",
            "-f",
            safe_base,
            "-o",
            str(out_dir),
            str(series_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[FAIL] dcm2niix failed: {series_dir}")
        print(result.stdout)
        print(result.stderr)
        continue  # ★ここで次へ進む

    nifti_path = out_dir / (f"{safe_base}.nii.gz" if GZ else f"{safe_base}.nii")
    if EXTRACT_FIRST_VOL and nifti_path.exists():
        try:
            extract_first_volume_inplace(nifti_path)
        except Exception as e:
            print(f"[WARN] failed to extract first vol: {nifti_path.name} ({e})")
