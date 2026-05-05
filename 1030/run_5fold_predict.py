from pathlib import Path
import json, shutil, subprocess

# === ここをあなたの環境に合わせる ===
dataset_id = 1
config = "3d_fullres"

nnUNet_raw = Path(r"C:\Users\orilab\Desktop\masumoto\1030\nnUNet_raw")
nnUNet_results = Path(r"C:\Users\orilab\Desktop\masumoto\1030\nnUNet_results")

dataset_name = "Dataset001_lumber"
imagesTr = nnUNet_raw / dataset_name / "imagesTr"

trainer_dir = nnUNet_results / dataset_name / f"nnUNetTrainer__nnUNetPlans__{config}"
splits_json = trainer_dir / "fold_0" / "splits_final.json"  # どのfoldでも同じ内容

oof_out = Path(r"C:\Users\orilab\Desktop\masumoto\1030\oof_pred")
oof_out.mkdir(parents=True, exist_ok=True)

with open(splits_json, "r", encoding="utf-8") as f:
    splits = json.load(f)


def copy_val_images(case_ids, tmp_in: Path):
    tmp_in.mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        # nnU-Netのcase idは "case001" 形式（あなたの現状に合わせている）
        src = tmp_in / f"{cid}_0000.nii.gz"
        if not src.exists():
            # imagesTr 内の該当ファイルを探す（1ch想定）
            found = list(imagesTr.glob(f"{cid}_0000.nii.gz"))
            if not found:
                raise FileNotFoundError(f"Missing image for {cid} in {imagesTr}")
            shutil.copy2(found[0], tmp_in / found[0].name)


for fold in range(5):
    val_cases = splits[fold]["val"]
    tmp_in = oof_out / f"_tmp_in_fold{fold}"
    tmp_out = oof_out / f"_tmp_out_fold{fold}"
    if tmp_in.exists():
        shutil.rmtree(tmp_in)
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    copy_val_images(val_cases, tmp_in)

    cmd = [
        "nnUNetv2_predict",
        "-d",
        str(dataset_id),
        "-c",
        config,
        "-i",
        str(tmp_in),
        "-o",
        str(tmp_out),
        "-f",
        str(fold),
        "-chk",
        "checkpoint_best.pth",
    ]
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd, shell=True)

    # fold出力を最終oof_out直下へ集約（重複しない：valはfoldごとに別だから）
    for p in tmp_out.glob("*.nii.gz"):
        shutil.copy2(p, oof_out / p.name)

    shutil.rmtree(tmp_in, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)

print("DONE. OOF predictions saved to:", oof_out)
