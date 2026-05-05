from pathlib import Path
import nibabel as nib
import numpy as np
import csv
from collections import defaultdict

# ===== 設定 =====
IMAGES_TR = Path(r"C:\Users\orilab\Desktop\masumoto\2dunet\Dataset\imagesTr")
NIFTI_POOL = Path(r"C:\Users\orilab\Desktop\nifti_folder")
OUT_CSV = Path(
    r"C:\Users\orilab\Desktop\masumoto\2dunet\case_to_candidates_top3_fp.csv"
)

TOPK = 3
FP_TARGET = (64, 64, 32)  # fingerprint解像度


def iter_nifti(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.name.endswith((".nii", ".nii.gz")):
            yield p


def load_vol0(path: Path) -> np.ndarray:
    """3D前提。4Dなら[...,0]。RGB等は除外。"""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.dtype.kind not in ("f", "i", "u"):
        raise TypeError(f"non-numeric dtype: {data.dtype}")
    return data.astype(np.float32, copy=False)


def stats_3d(vol: np.ndarray):
    return float(vol.min()), float(vol.max()), float(vol.mean())


def fingerprint3d(vol: np.ndarray, target=FP_TARGET) -> np.ndarray:
    """粗視化3D + 分位点正規化 → 1Dベクトル"""
    x, y, z = vol.shape
    tx, ty, tz = target

    xs = np.linspace(0, x - 1, tx).astype(int)
    ys = np.linspace(0, y - 1, ty).astype(int)
    zs = np.linspace(0, z - 1, tz).astype(int)

    small = vol[np.ix_(xs, ys, zs)].astype(np.float32, copy=False)

    lo = np.quantile(small, 0.02)
    hi = np.quantile(small, 0.98)
    if hi - lo < 1e-6:
        small = small - lo
    else:
        small = np.clip(small, lo, hi)
        small = (small - lo) / (hi - lo)

    return small.reshape(-1)


def fp_distance(a: np.ndarray, b: np.ndarray) -> float:
    """小さいほど似てる"""
    return float(np.mean((a - b) ** 2))


# ===== 1) pool を shape 別に index（stats + fingerprint を保存） =====
pool_by_shape = defaultdict(list)

print("Indexing nifti_folder (pool)...")
for p in iter_nifti(NIFTI_POOL):
    try:
        vol = load_vol0(p)
        fp = fingerprint3d(vol)
    except Exception:
        continue

    shp = vol.shape
    st = stats_3d(vol)
    pool_by_shape[shp].append((p, st, fp))

print("Pool shapes:", len(pool_by_shape))

# ===== 2) imagesTr 各caseについて topK =====
rows = []
cases = sorted(
    [p for p in iter_nifti(IMAGES_TR) if p.name.endswith(("_0000.nii.gz", "_0000.nii"))]
)
print(f"Processing cases: {len(cases)}")

for case_path in cases:
    try:
        case_vol = load_vol0(case_path)
        case_fp = fingerprint3d(case_vol)
    except Exception as e:
        rows.append(
            [
                str(case_path),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "case_load_failed",
            ]
        )
        continue

    shp = case_vol.shape
    candidates = pool_by_shape.get(shp, [])
    if not candidates:
        rows.append(
            [
                str(case_path),
                str(shp),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "no_pool_same_shape",
            ]
        )
        continue

    case_st = stats_3d(case_vol)

    scored = []
    for cand_path, cand_st, cand_fp in candidates:
        dist = fp_distance(case_fp, cand_fp)
        scored.append((dist, cand_path, cand_st))

    scored.sort(key=lambda x: x[0])
    top = scored[:TOPK]

    out = [
        str(case_path),
        str(shp),
        f"min={case_st[0]:.1f},max={case_st[1]:.1f},mean={case_st[2]:.3f}",
    ]

    for rank, (dist, cand_path, cand_st) in enumerate(top, start=1):
        out += [
            cand_path.name,
            f"{dist:.6f}",  # fpdist
            f"min={cand_st[0]:.1f},max={cand_st[1]:.1f},mean={cand_st[2]:.3f}",
        ]

    while len(out) < 3 + TOPK * 3:
        out += ["", "", ""]

    out.append("ok")
    rows.append(out)

# ===== 3) CSV =====
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

header = ["case_path", "shape", "case_stats"]
for i in range(1, TOPK + 1):
    header += [f"cand{i}", f"fpdist{i}", f"cand{i}_stats"]
header += ["status"]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as wf:
    w = csv.writer(wf)
    w.writerow(header)
    w.writerows(rows)

print(f"Done. Wrote: {OUT_CSV}")
