# DeepLearning

MRI（DTI/DWI）の3D医療画像を対象に、**脊椎神経根（nerve root）** と **硬膜管（dural tube）** の自動セグメンテーションを研究・実験したフォルダです。  
PyTorch Lightning / segmentation_models_pytorch / nnUNet など複数のフレームワークを使い、さまざまなアプローチを比較しています。

---

## フォルダ構成の全体像

```
masumoto/
│
├── pytorchLightning/          # ★現行メイン: 3D Attention U-Net (PyTorch Lightning)
├── pytorchLightning_Tanaka/   # 田中さん用の 3D U-Net コピー
├── 2dunet/                    # 2D/3D U-Net の多様な実験（多数のバリアント）
├── MajorityAlgorhythm/        # 多数決アンサンブル (Axial/Coronal/Sagittal 3方向)
├── PL_2D/                     # PyTorch Lightning による 2D セグメンテーション
├── smp/                       # segmentation_models_pytorch の実験
├── 1030/                      # nnUNet (ResEnc) による 5-fold 実験
├── nnUNet/                    # nnUNet v2 ソースコード（git clone）
│
├── tractography/              # DIPy による DTI ファイバートラクトグラフィー
├── checkVector/               # DTI 固有ベクトル・FA 値の可視化・確認
├── voxelMorph/                # VoxelMorph による医療画像位置合わせ
│
├── MRIcron/                   # MRIcron 画像ビューアアプリ
├── namedFAs/                  # 患者ごとの FA マップ (.nii.gz)
├── assigned_nifti/            # 患者 ID 対応付き NIfTI データ
├── 0918/, 1002/               # 症例ごとの生 DTI/DWI データ
│
├── get_fa_3.py                # FA 値の左右比較スクリプト（ルートレベル）
├── 0916DTI.nii 等             # ルートレベルの DTI/FA ファイル
└── README.md                  # このファイル
```

---

## フォルダ別・ファイル別の役割

---

### [`pytorchLightning/`](pytorchLightning/) — 現行メイン実装 (3D Attention U-Net)

| ファイル | 役割 |
|---|---|
| [`dataset.py`](pytorchLightning/dataset.py) | NIfTI 3D 画像を読み込み、`[0,1]`正規化・リサイズして PyTorch テンソルを返す `NiftiDataset` |
| [`dataModule.py`](pytorchLightning/dataModule.py) | fold ファイルを使って train/val/test に分割し、`DataLoader` を管理する `DataModule` |
| [`model.py`](pytorchLightning/model.py) | U-Net + EfficientNet-B0 の基本 3D モデル（初期実装） |
| [`attention_model.py`](pytorchLightning/attention_model.py) | **現行メインモデル**：MONAI の 3D Attention U-Net + クラス重み付き Dice+CE 複合損失。クラス重み: bg=0.1 / nerve=1.5 / dural=0.5。テスト時に予測/GT/入力画像を `.nii.gz` で保存し、クラスごとの Dice も出力 |
| [`train.py`](pytorchLightning/train.py) | 基本モデルの学習スクリプト（100 epoch / EfficientNet-B0） |
| [`train_attantion.py`](pytorchLightning/train_attantion.py) | Attention U-Net の学習スクリプト（300 epoch / AdamW lr=1e-4 / チェックポイント付き） |
| [`test.py`](pytorchLightning/test.py) | チェックポイントを読み込み Attention U-Net でテストを実行 |
| [`inference.py`](pytorchLightning/inference.py) | 2 つの NIfTI ファイル間のクラスごと Dice 係数を計算・表示（単発評価用） |
| [`pred_2D.py`](pytorchLightning/pred_2D.py) | 3D NIfTI → 2D PNG スライスに変換し、2D U-Net で推論後に 3D NIfTI に再構成して保存 |
| [`change_value.py`](pytorchLightning/change_value.py) | マスクのラベル値 1↔2 を入れ替えるユーティリティ（アノテーションミス修正用） |
| [`1204.ps1`](pytorchLightning/1204.ps1) | `train.py` → `train_attantion.py` を順番に実行する PowerShell 起動スクリプト |

**サブディレクトリ:**

```
pytorchLightning/
├── data/
│   ├── images/         # 入力NIfTI (.nii) — 症例番号形式（00001.nii等）
│   ├── masks/          # セグメンテーションマスク (.nii.gz)
│   ├── split/          # fold_N.txt: train/val/test 症例番号リスト
│   │     例) train: 00001, 00002, 00004, ... (24症例)
│   │         val:   00028, 00029, 00030, ... (8症例)
│   │         test:  00003, 00006, 00010, ... (8症例)
│   └── test/           # テスト用データ
├── Niftis/             # 移行元NIfTIデータ（.nii/.nii.gz 混在）
├── 移行データ/          # 旧環境からコピーしたNIfTIデータ
├── checkpoints/        # 学習済みモデル (.ckpt)
├── logs/               # TensorBoard ログ
└── nifti_predictions/  # テスト時の予測マスク出力 (.nii.gz)
```

---

### [`pytorchLightning_Tanaka/`](pytorchLightning_Tanaka/) — 田中さん用コピー

`pytorchLightning/` とほぼ同じ構成（`dataset.py`, `dataModule.py`, `model.py`, `train.py`）。  
「ニセイのデータを継承した3D用フォルダ」とメモあり。田中さんの環境（`Tanaka/PytorchLightning/.lightningenv`）での再現実験向けに分離されたコピー。

---

### [`2dunet/`](2dunet/) — 2D/3D U-Net の多様な実験

実験・比較が多く、ファイル数が最も多いフォルダ。

#### 学習スクリプト

| ファイル | 内容 |
|---|---|
| [`train.py`](2dunet/train.py) | 2D U-Net（resnet34 エンコーダ）を PNG 画像で学習。クラス重み付き DiceLoss + F1 指標。 |
| [`segmentation.py`](2dunet/segmentation.py) | Google Colab から移植した初期実験スクリプト（segmentation_models_pytorch 使用） |
| [`train_3d.py`](2dunet/train_3d.py) | 自作 3D U-Net（NIfTI ボリューム単位。train/val/test または K-fold CV に対応） |
| [`train_3d_crop.py`](2dunet/train_3d_crop.py) | 3D U-Net + 空間クロップ版 |
| [`train_3d_aug_crop.py`](2dunet/train_3d_aug_crop.py) | 3D U-Net + データ拡張版 |
| [`train_3d_cubic.py`](2dunet/train_3d_cubic.py) | 3D U-Net + キュービック補間版 |
| [`train_3d_cross.py`](2dunet/train_3d_cross.py) | 3D U-Net + K-fold 交差検証 |
| [`train_3d_multi.py`](2dunet/train_3d_multi.py) | 神経根＋硬膜管のマルチタスク 3D U-Net |
| [`train_3d_multi_augument.py`](2dunet/train_3d_multi_augument.py) | マルチタスク + データ拡張 |
| [`train_all.py`](2dunet/train_all.py) | 全データを使った学習（val なし） |
| [`multi_cv.py`](2dunet/multi_cv.py) / [`multi_cubic_train.py`](2dunet/multi_cubic_train.py) | マルチタスク系の CV・キュービック補間版 |

#### テスト・評価スクリプト

| ファイル | 内容 |
|---|---|
| [`test_2d.py`](2dunet/test_2d.py) | 2D モデルのテスト |
| [`test_3d.py`](2dunet/test_3d.py) | 3D モデルのテスト |
| [`test_crop.py`](2dunet/test_crop.py) | クロップ版 3D のテスト |
| [`test_cubic.py`](2dunet/test_cubic.py) | キュービック補間版のテスト |
| [`test_multi.py`](2dunet/test_multi.py) | マルチタスク版のテスト |
| [`test_multi_aug.py`](2dunet/test_multi_aug.py) | マルチタスク + 拡張版のテスト |
| [`test_multi_cubic.py`](2dunet/test_multi_cubic.py) | マルチタスク + キュービック補間のテスト |
| [`test_best.py`](2dunet/test_best.py) / [`test_only.py`](2dunet/test_only.py) | ベストモデル評価・テスト専用 |
| [`compare.py`](2dunet/compare.py) | **統計比較ツール**：各条件の `test_metrics.csv` を収集し、Friedman 検定 + Wilcoxon 符号順位検定（Holm 補正）で条件間の有意差を検証 |

#### ユーティリティ

| ファイル | 内容 |
|---|---|
| [`dicom_to_nifti.py`](2dunet/dicom_to_nifti.py) | DICOM フォルダを再帰的に探索し、`dcm2niix` で NIfTI に一括変換。4D なら最初のボリュームだけ抽出 |
| [`convert.py`](2dunet/convert.py) | `.nii` → `.nii.gz` に変換 |
| [`crop_cv.py`](2dunet/crop_cv.py) | クロップ付き K-fold CV |
| [`nifti_check.py`](2dunet/nifti_check.py) | NIfTI の「フィンガープリント（粗視化3Dベクトル）」で類似ファイルを MSE 距離でマッチングし TOP3 を CSV 出力 |
| [`nifti_folder_make.py`](2dunet/nifti_folder_make.py) | ネスト構造の NIfTI をフラットフォルダに統合（同名ファイルは連番リネーム） |
| [`roi_check.py`](2dunet/roi_check.py) | マスク内ラベル1の X/Y/Z バウンディングボックスを一覧表示・CSV 保存 |
| [`pred_nifti.py`](2dunet/pred_nifti.py) | 学習済みマルチタスク 3D U-Net での推論。神経根・硬膜管の予測マスクを NIfTI で保存 |

#### 実行スクリプト

| ファイル | 役割 |
|---|---|
| [`run_all.ps1`](2dunet/run_all.ps1) | アブレーション実験 `exp06〜09`（等方化・パッチ・z-スコアの有無の組み合わせ）を連続実行 |
| [`run0119.ps1`](2dunet/run0119.ps1) | 2D/3D 各モデルを特定条件で実行する PowerShell スクリプト |
| [`1211.sh`](2dunet/1211.sh) | Bash 版の一括実行スクリプト（WSL/Linux 環境向け） |

**サブディレクトリ:**

```
2dunet/
├── Dataset/                    # nnUNet形式データセット（dataset.json付き）
│   ├── dataset.json            # {"name":"lumber", labels:{bg:0, class1:1, class2:2}, numTraining:23}
│   ├── imagesTr/ labelsTr/     # 学習用 NIfTI (.nii.gz)
│   ├── imagesTs/ labelsTs/     # テスト用 NIfTI
│   └── train/ val/ test/       # 分割済みデータ（image/mask サブフォルダ）
├── Dataset001_lumber/          # nnUNet Dataset001 形式（学習・テスト分割）
│   ├── imagesTr/ labelsTr/     # 学習用: case001_0000.nii.gz 形式
│   └── imagesTs/ labelsTs/     # テスト用
├── Dataset001_lumber_split/    # train/val/test の3分割版 Dataset001
│   ├── imagesTr/ labelsTr/
│   ├── imagesVa/ labelsVa/
│   └── imagesTs/ labelsTs/
├── data/                       # train/val/test + image/mask 構成の生データ
├── ckpt3d/ ckpt3d-*-crop/ 等  # 各条件の学習済みモデル
├── pred3d/ pred3d-*-crop/ 等  # 各モデルの予測結果
├── ckpt3d_mt/ pred3d_mt/ 等   # マルチタスクモデルの重み・予測
└── runs/                       # 統計比較用 test_metrics.csv 格納フォルダ
```

---

### [`MajorityAlgorhythm/`](MajorityAlgorhythm/) — 多数決アンサンブル (3方向)

Axial / Coronal / Sagittal の 3 方向で別々に 2D U-Net を学習し、多数決で最終予測を決定するアプローチ。

| ファイル | 役割 |
|---|---|
| [`dataset.py`](MajorityAlgorhythm/dataset.py) | 複数の症例フォルダから PNG を収集し、マスクリマッピング（0/127/255 → 0/1/2）を行う `PNGDataset` |
| [`dataModule.py`](MajorityAlgorhythm/dataModule.py) | fold 分割対応の DataModule。断面方向（Axial/Coronal/Sagittal）を指定して対応フォルダを自動選択 |
| [`split.py`](MajorityAlgorhythm/split.py) | NIfTI を3方向スライスに変換して PNG として保存する前処理。Coronal・Sagittal は等方性補正のため 2.4 倍にリサイズ |
| [`model_2D.py`](MajorityAlgorhythm/model_2D.py) | 2D U-Net (EfficientNet-B0) + DiceLoss + CrossEntropyLoss。テスト時に NIfTI 保存 |
| [`train_2D.py`](MajorityAlgorhythm/train_2D.py) | 学習スクリプト（200 epoch / mixed precision） |
| [`test_2D.py`](MajorityAlgorhythm/test_2D.py) | テストスクリプト |
| [`check.py`](MajorityAlgorhythm/check.py) | CUDA / GPU 動作確認（PyTorch・CUDA バージョン・GPU 名を出力） |

**サブディレクトリ:**

```
MajorityAlgorhythm/
├── pngData/
│   ├── images/
│   │   ├── 00001/ 〜 00005/ ...  # 症例ごとのPNG（Axial/Coronal/Sagittal）
│   ├── masks/
│   └── split/          # fold分割情報
├── checkpoints/        # 学習済みモデル
└── logs/               # TensorBoardログ
```

---

### [`PL_2D/`](PL_2D/) — PyTorch Lightning による 2D セグメンテーション

NIfTI を PNG スライスに変換してから 2D U-Net で学習するパイプライン。

| ファイル | 役割 |
|---|---|
| [`nifti_to_png.py`](PL_2D/nifti_to_png.py) | NIfTI 3D 画像をアキシャルスライスごとに PNG 変換。マスクは 0/127/255 の固定値にマッピング |
| [`dataset_2D.py`](PL_2D/dataset_2D.py) | PNG 画像ペアを読み込む 2D データセット |
| [`dataModule_2D.py`](PL_2D/dataModule_2D.py) | fold 分割対応の 2D DataModule |
| [`model_2D.py`](PL_2D/model_2D.py) | 2D U-Net (EfficientNet-B0) Lightning モジュール（DiceLoss） |
| [`train_2D.py`](PL_2D/train_2D.py) | 学習スクリプト |
| [`test_2D.py`](PL_2D/test_2D.py) | テストスクリプト |
| [`predict_2D.py`](PL_2D/predict_2D.py) | 新規データへの推論 |
| [`save_predictions.py`](PL_2D/save_predictions.py) | 予測結果を NIfTI に保存 |

**サブディレクトリ:**

```
PL_2D/
├── png_data/           # 変換済みPNGデータ（images/masks）
├── png_data0206/       # 2月6日版PNGデータ（images/masks）
├── temp/               # 作業中・一時ファイル
└── predictions_1223_2D/# 2D推論結果（12月23日版）
```

---

### [`smp/`](smp/) — segmentation_models_pytorch の直接実験

PyTorch Lightning を使わず、`smp.utils.train` の `TrainEpoch` / `ValidEpoch` で学習する実験フォルダ。

| ファイル | 役割 |
|---|---|
| [`train.py`](smp/train.py) | NIfTI スライスから直接学習する 2D U-Net（resnet34 / 200 epoch） |
| [`train_attantion.py`](smp/train_attantion.py) | Attention U-Net バリアント（NIfTI 入力） |
| [`train_attantion_png.py`](smp/train_attantion_png.py) | Attention U-Net（PNG 入力） |
| [`train_png.py`](smp/train_png.py) | PNG 入力の通常 U-Net |
| [`test.py`](smp/test.py) | テスト・Dice 評価 |
| [`test_attantion.py`](smp/test_attantion.py) / [`test_attantion_png.py`](smp/test_attantion_png.py) | Attention モデルのテスト |
| [`predict_nif.py`](smp/predict_nif.py) | NIfTI 形式での推論 |
| [`predict_png.py`](smp/predict_png.py) | PNG での推論 |
| [`data_rotate.py`](smp/data_rotate.py) | データ回転拡張 |
| [`clasify_weight.py`](smp/clasify_weight.py) | クラス重み計算 |
| [`unique.py`](smp/unique.py) | マスクのユニーク値確認 |
| [`segmentation.ipynb`](smp/segmentation.ipynb) | 初期実験用 Jupyter ノートブック |

**サブディレクトリ:**

```
smp/
├── Dataset001_lumber/  # nnUNet形式データ（imagesTr/labelsTr/imagesTs/labelsTs）
├── data2/              # train/val/test + image/mask 構成
│   └── train/ val/ test/
│       ├── image/
│       └── mask/
└── nifti_data/         # 生NIfTIデータ（images/masks）
```

---

### [`1030/`](1030/) — nnUNet (ResEnc) 5-fold 実験

nnUNet v2 の ResEncoder モデルを用いた 5-fold 交差検証の実験フォルダ。

| ファイル | 役割 |
|---|---|
| [`make.py`](1030/make.py) | マスク内の1ボクセルをラベル2に書き換えるデバッグ用スクリプト（評価パイプラインのクラス2認識確認用） |
| [`metrics_test.py`](1030/metrics_test.py) | GT と予測 NIfTI を症例ごとに比較し、**Dice / HD95（mm）/ ASD（mm）/ Boundary IoU** を計算して CSV 保存。NIfTI スペーシングを使った実寸法計算に対応 |
| [`run_5fold_predict.py`](1030/run_5fold_predict.py) | nnUNet の `splits_final.json` を読み込み、5-fold の OOF 予測を `nnUNetv2_predict` コマンドで自動実行 |
| [`filename.py`](1030/filename.py) | ファイル名操作ユーティリティ |

**5-fold 平均 Dice（神経根）の結果:**

| Fold | Dice |
|---|---|
| 0 | 0.705 |
| 1 | 0.714 |
| 2 | 0.701 |
| 3 | 0.701 |
| 4 | 0.698 |

**サブディレクトリ:**

```
1030/
├── Dataset001_lumber/          # nnUNet Dataset001 形式
├── nnUNet_raw/                 # nnUNet生データ格納先
├── nnUNet_preprocessed/        # nnUNet前処理済みデータ
├── nnUNet_results/             # nnUNet学習結果
├── nnUNet/                     # nnUNet ソースコード（git clone）
├── pred_out/                   # 予測出力（通常モデル）
├── pred_out_fold0〜4/          # fold別予測出力
├── pred_out_resenc_fold0〜3/   # ResEnc版 fold別予測出力
├── pred_out_resenc_ens_f0-3/   # ResEnc アンサンブル（fold0-3）
├── pred_out_ens_0to4_best/     # 通常モデル 全fold アンサンブル
├── nnUNet_fold0-4_result/      # 全fold結果まとめ
└── temp/                       # 作業中・一時ファイル
```

**評価結果ファイル:**

| ファイル | 内容 |
|---|---|
| `metrics_resenc_fold1〜3_test.csv` | ResEnc 各fold テスト評価 |
| `metrics_resenc_ens_f0-3_test.csv` | ResEnc アンサンブル評価 |
| `dice_report_resenc_fold0.csv/.json` | fold0 Dice詳細レポート |

---

### [`nnUNet/`](nnUNet/) — nnUNet v2 ソースコード

```
nnUNet/
└── nnUNet/         # nnUNet リポジトリ（多重クローン構造）
    └── nnunetv2/   # nnUNet v2 Pythonパッケージ本体
```

---

### [`tractography/`](tractography/) — DTI ファイバートラクトグラフィー

| ファイル | 役割 |
|---|---|
| [`dipy.py`](tractography/dipy.py) | DIPy を使い、DWI データから DTI テンソルモデルを構築し局所トラッキングでファイバーを生成 |
| [`check_DWI_mask.py`](tractography/check_DWI_mask.py) | 脳マスク NIfTI のボクセル座標を Plotly で3D散布図にして可視化 |
| `iida_dcm2niix.*` | サンプル症例の DWI データ（.nii.gz / .bval / .bvec） |

---

### [`checkVector/`](checkVector/) — DTI 固有ベクトル・FA 値の検証

| ファイル | 役割 |
|---|---|
| [`FA_vector_check.py`](checkVector/FA_vector_check.py) | FA の固有ベクトル（V1）と固有値（L1）を特定ボクセルで確認し、矢印プロットと熱マップで2D可視化 |
| [`tractography.py`](checkVector/tractography.py) | ROI 内の DTI 固有ベクトル（V1）を z=24 と z=31 の2スライスで3D矢印プロット（Matplotlib）で可視化 |

**サブディレクトリ:**

```
checkVector/
└── yamadaYouko/    # 山田洋子患者のDTI解析データ
    ├── Only2ROI.nii.gz / Only3ROI.nii(.gz)  # ROIマスク
    ├── Yamada_FA.nii.gz                       # FA map
    ├── Yamada_L1〜L3.nii(.gz)                 # 固有値
    └── (その他 DTI metrics)
```

---

### [`voxelMorph/`](voxelMorph/) — 医療画像位置合わせ

| ファイル | 役割 |
|---|---|
| [`voxelMorph.ipynb`](voxelMorph/voxelMorph.ipynb) | VoxelMorph を用いた 3D 医療画像の非剛体位置合わせ実験 Jupyter ノートブック |
| `my_data.npy` | 変換用データ（NumPy形式） |
| `IMG_0713.mov` | 参照動画（実験記録） |

**サブディレクトリ:**

```
voxelMorph/
└── frames/     # 動画から抽出した連番フレーム画像 (frame_0000.png 〜)
```

---

### [`1002/`](1002/) — 患者DWIデータ・DTI解析結果

```
1002/
├── ニセイ/         # ニセイグループ患者データ
└── リラ/           # リラグループ患者データ（26名）
    ├── namedPeople/  # 患者ごとのDTIデータフォルダ
    │   └── [患者名_症例番号]/     # 例: 0111Kamada_28144/
    │       ├── dwi_images/         # DWI画像（下記参照）
    │       ├── [患者名].bval/.bvec # 拡散強度・方向情報
    │       ├── [患者名].nii.gz     # 元DWI
    │       ├── [患者名]_FA.nii.gz  # FA map
    │       ├── [患者名]_L1〜L3.nii.gz  # 固有値
    │       ├── [患者名]_V1〜V3.nii.gz  # 固有ベクトル
    │       ├── [患者名]_brain.nii.gz / _brain_mask.nii.gz  # 脳抽出
    │       ├── [患者名]_ecc.nii.gz # 渦電流補正済みDWI
    │       ├── dti.trk / res.trk   # 繊維追跡結果（.trk形式）
    │       └── roi.nii             # ROIマスク
    ├── outputFAs/
    ├── faKamada/ faYamadaYouko/    # 個別FA解析
    └── [その他患者フォルダ]
```

**dwi_images/ の構造（各患者共通）:**

```
dwi_images/
├── x/              # X方向スライス
├── y/              # Y方向スライス
├── z/              # Z方向スライス（元データ）
├── z_extracted/    # 抽出済みスライス
└── z_filled/
    ├── image/      # 補完済み画像
    └── mask/       # 補完済みマスク
```

---

### [`0918/`](0918/) — 初期患者DWIデータ（raw）

```
0918/
├── zaitsu/         # zaitsu患者: 0111_dcm2niix.{bval,bvec,json,nii.gz,txt}
└── 0214takahashi/  # takahashi患者: dcm2niix変換後ファイル
```

---

### [`assigned_nifti/`](assigned_nifti/) — 患者IDで整理されたNIfTIデータ

```
assigned_nifti/
├── 0006041058/     # 患者IDごとのNIfTIファイル（例: case034_0000.nii.gz）
├── 0007100648/
├── 0008893374/
├── 0010130038/
├── 0010253999/
└── mapping_folder2_to_patientid.csv  # フォルダ名↔患者IDのマッピング表
```

---

### ルートレベルのスクリプト・ファイル

| ファイル | 役割 |
|---|---|
| [`get_fa_3.py`](get_fa_3.py) | ROI マスクから神経根領域を左右に分けて、各スライスごとに FA 値の平均・差分・フラグ（閾値0.1）を計算し CSV に保存 |
| `namedFAs/` | 各患者の FA マップ（`姓_FA.nii.gz`、18 名分） |
| `0916DTI.nii`, `0916DTINII.nii.gz` | 9 月 16 日取得の DTI データ |
| `0916BVAL.bval`, `0916BVEC.bvec` | 対応する拡散方向情報 |
| `DWI_FA_normalized.nii` | 正規化済み FA マップ |
| `case_to_candidates_top3.csv` | 症例候補 TOP3 の対応 CSV |
| `debug_mismatch.csv` | データ不一致のデバッグログ |
| `MRIcron/` | MRIcron 画像ビューアアプリケーション |
| `MicroDicom-2024.2-x64.exe` | DICOM ビューアインストーラ |
| `itksnap-4.2.2-20241202-win64-AMD64.exe` | ITK-SNAP インストーラ（NIfTI 閲覧・アノテーション） |
| `移行データ-*.zip` | 旧環境からの移行データアーカイブ |

---

## データの流れ（全体）

```
DICOM データ
    │
    ▼ dicom_to_nifti.py
NIfTI 3D データ（.nii.gz）
    │
    ├─── [3D モデル系]
    │        │
    │        ├── pytorchLightning/  → 3D Attention U-Net → nifti_predictions-*/
    │        ├── 2dunet/train_3d*.py → 3D 自作 U-Net → pred3d*/
    │        └── 1030/ (nnUNet)    → pred_out_resenc_fold*/
    │
    └─── [2D モデル系]
             │
             ├── nifti_to_png.py または smp/train.py（スライス単位）
             ▼
          2D PNG スライス
             │
             ├── 2dunet/train.py       → 2D U-Net (PNG)
             ├── MajorityAlgorhythm/   → 3方向アンサンブル
             ├── PL_2D/                → PL ベース 2D
             └── smp/                  → smp ベース 2D
```

---

## セグメンテーションクラス

| ラベル値 | クラス名 |
|---|---|
| 0 | 背景（background） |
| 1 | 神経根（nerve root） |
| 2 | 硬膜管（dural tube / spinal） |

---

## 環境情報

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080（VRAM 16 GB） |
| CUDA | 12.6 |
| OS | Windows 11 |
| Python 環境 | 仮想環境ごとに分離（`.lightningenv`, `.smpenv`, `.nnenv` 等） |
| 主要ライブラリ | PyTorch Lightning, segmentation_models_pytorch, segmentation_models_pytorch_3d, MONAI, nibabel, DIPy, nnUNetv2 |
