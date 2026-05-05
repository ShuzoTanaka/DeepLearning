# run_ablation_4runs.ps1
# 4条件を連続で: train -> test -> train -> test ...
# それぞれ out_dir を変えて結果が上書きされないようにする

# ====== あなたの環境に合わせて編集 ======
$Python = (Get-Command python).Source          # もしくは "C:\...\python.exe"
$TrainPy = "train_all.py"       # train.pyのファイル名
$DatasetRoot = "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset"  # nnU-Net風 Dataset ルート
# =======================================

# 共通ハイパラ（必要なら調整）
$epochs = 200
$batch_size = 1
$lr = 1e-3
$val_ratio = 0.2
$num_workers = 2

# loss は "複合ロス" = multitask 固定
$loss_mode = "multitask"
$lambda_root = 1.0
$lambda_dura = 0.3

# crop（train.pyのデフォルトと同じなら省略可）
$crop_x = "50 200"
$crop_y = "45 210"

# パッチ設定（必要なら変える）
$patch_size = "48 192 224"
$fg_ratio = 0.33

# 等方化 spacing（enable_isotropic のときのみ使われる）
$target_spacing = "1.0 1.0 1.0"

function Run-Exp {
    param(
        [string]$Name,
        [bool]$EnableIsotropic,
        [bool]$EnablePatch,
        [bool]$EnableZscore
    )

    $outDir = Join-Path -Path "." -ChildPath ("runs\" + $Name)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $args = @(
        $TrainPy,
        "--dataset_root", $DatasetRoot,
        "--epochs", $epochs,
        "--batch_size", $batch_size,
        "--lr", $lr,
        "--val_ratio", $val_ratio,
        "--num_workers", $num_workers,
        "--out_dir", $outDir,
        "--save_name", "best",
        "--loss_mode", $loss_mode,
        "--lambda_root", $lambda_root,
        "--lambda_dura", $lambda_dura,
        "--enable_augment",
        "--crop_x", $crop_x.Split(" "),
        "--crop_y", $crop_y.Split(" ")
    )

    # ★ zscore を使うときだけ付与
    if ($EnableZscore) {
        $args += @("--enable_zscore_norm")
    }

    if ($EnableIsotropic) {
        $args += @("--enable_isotropic", "--target_spacing_mm")
        $args += $target_spacing.Split(" ")
    }

    if ($EnablePatch) {
        $args += @("--enable_patch", "--patch_size")
        $args += $patch_size.Split(" ")
        $args += @("--fg_ratio", $fg_ratio)
    }

    Write-Host "============================================================"
    Write-Host "RUN: $Name"
    Write-Host "  isotropic = $EnableIsotropic, patch = $EnablePatch, zscore = $EnableZscore, augment = True, loss = multitask"
    Write-Host "  out_dir = $outDir"
    Write-Host "============================================================"

    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $Name (exit code = $LASTEXITCODE)"
    }

    $csv = Join-Path $outDir "test_metrics.csv"
    if (Test-Path $csv) {
        Write-Host "OK: saved $csv"
    }
    else {
        Write-Host "WARN: test_metrics.csv not found (imagesTs/labelsTs が無い可能性)"
    }
}


# （追加で回す）等方化なし、パッチあり、zscoreなし、augment、複合ロス
Run-Exp -Name "exp06_noIso_patch_noZscore_aug_multitask" -EnableIsotropic $false -EnablePatch $true  -EnableZscore $false

# （追加で回す）等方化あり、パッチなし、zscoreなし、augment、複合ロス
Run-Exp -Name "exp07_Iso_noPatch_noZscore_aug_multitask" -EnableIsotropic $true  -EnablePatch $false -EnableZscore $false

# （追加で回す）等方化あり、パッチあり、zscoreなし、augment、複合ロス
Run-Exp -Name "exp08_Iso_patch_noZscore_aug_multitask" -EnableIsotropic $true  -EnablePatch $true  -EnableZscore $false

# （追加で回す）等方化あり、パッチなし、zscoreあり、augment、複合ロス
Run-Exp -Name "exp09_Iso_noPatch_zscore_aug_multitask" -EnableIsotropic $true  -EnablePatch $false -EnableZscore $true

Write-Host "ALL DONE."
