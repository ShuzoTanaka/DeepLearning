ckpt3d 学習率 0.001

case003: 0.7508
case006: 0.6768
case010: 0.0000
case011: 0.7264
case036: 0.6322
case037: 0.6331
case039: 0.7328
case040: 0.3740
Mean Dice (nerve root): 0.5658

ckpt3d-4-100 学習率 0.0001 100epoch

=== Test Dice (nerve root only) ===
case003: 0.6857
case006: 0.6254
case010: 0.0137
case011: 0.6893
case036: 0.4264
case037: 0.5594
case039: 0.6840
case040: 0.2119
Mean Dice (nerve root): 0.4870

ckpt3d-4-1000 学習率 0.0001 1000epoch

=== Test Dice (nerve root only) ===
case003: 0.6763
case006: 0.6159
case010: 0.0680
case011: 0.6791
case036: 0.4805
case037: 0.5732
case039: 0.6860
case040: 0.2401
Mean Dice (nerve root): 0.5024

python train_3d_crop.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-3-crop"

<!-- 精度上がってるかも！ -->

=== Case-wise Dice (nerve root only, cropped) ===
case003.nii: 0.7666
case006.nii: 0.7108
case010.nii: 0.0000
case011.nii: 0.7159
case036.nii: 0.5416
case037.nii: 0.6345
case039.nii: 0.7325
case040.nii: 0.2885
Mean Dice (nerve root, cropped): 0.5488

python train_3d_crop.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-4 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-4-crop"

case003.nii: 0.7031
case006.nii: 0.6588
case010.nii: 0.0000
case011.nii: 0.6720
case036.nii: 0.0071
case037.nii: 0.4511
case039.nii: 0.5704
case040.nii: 0.1011
Mean Dice (nerve root, cropped): 0.3955

python train_3d_multi.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --nerve_root_label 1 --dura_label 2 --save_name "multitask_crop_lr1e3"

<!-- 精度上がってるかも！！ -->

ckpt = torch.load(best_path, map_location=device)
=== Case-wise Dice (nerve root only) ===
case003.nii: 0.7766
case006.nii: 0.7036
case010.nii: 0.0228
case011.nii: 0.7016
case036.nii: 0.4670
case037.nii: 0.5755
case039.nii: 0.6779
case040.nii: 0.4130
Mean Dice (nerve root): 0.5422

python train_3d_multi.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-4 --nerve_root_label 1 --dura_label 2 --save_name "multitask_crop_lr1e4"

=== Case-wise Dice (nerve root only) ===
case003.nii: 0.7309
case006.nii: 0.6724
case010.nii: 0.0000
case011.nii: 0.6707
case036.nii: 0.1012
case037.nii: 0.4401
case039.nii: 0.5900
case040.nii: 0.0327
Mean Dice (nerve root): 0.4048

train_3d_aug_crop.py(って書いてあるけど crop はしてなかったみたい) 0.001
--ckpt_path C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-aug-3-crop\best_3dunet_nerve.pth --nerve_root_label 1 --pred_dir C:\Users\orilab\Desktop\masumoto\2dunet\pred3d-3-aug
=== Case-wise Dice (nerve root only, cropped) ===
case003.nii: 0.6264
case006.nii: 0.4787
case010.nii: 0.3276
case011.nii: 0.6375
case036.nii: 0.4565
case037.nii: 0.4111
case039.nii: 0.5018
case040.nii: 0.2958
Mean Dice (nerve root, cropped): 0.4669

train_3d_aug_crop.py(って書いてあるけど crop はしてなかったみたい) 0.0001
--ckpt_path C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-aug-4-crop\best_3dunet_nerve.pth --nerve_root_label 1 --pred_dir C:\Users\orilab\Desktop\masumoto\2dunet\pred3d-4-aug
=== Case-wise Dice (nerve root only, cropped) ===
=== Case-wise Dice (nerve root only, cropped) ===
case003.nii: 0.7205
case006.nii: 0.5638
case010.nii: 0.0000
case011.nii: 0.6666
case036.nii: 0.4721
case037.nii: 0.4784
case039.nii: 0.6035
case040.nii: 0.1187
Mean Dice (nerve root, cropped): 0.4530

ckpt = torch.load(best_path, map_location=device)

python train_3d_multi_augument.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --nerve_root_label 1 --dura_label 2 --save_name "multitask_crop_aug_lr1e3"l
=== Case-wise Dice (nerve root only) ===
case003.nii: 0.7477
case006.nii: 0.7156
case010.nii: 0.0000
case011.nii: 0.7287
case036.nii: 0.5124
case037.nii: 0.6174
case039.nii: 0.7189
case040.nii: 0.3249
Mean Dice (nerve root): 0.5457

python train_3d_multi_augument.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --nerve_root_label 1 --dura_label 2 --save_name "multitask_crop_aug_lr1e3"
=== Case-wise Dice (nerve root only) ===
case003.nii: 0.7477
case006.nii: 0.7156
case010.nii: 0.0000
case011.nii: 0.7287
case036.nii: 0.5124
case037.nii: 0.6174
case039.nii: 0.7189
case040.nii: 0.3249
Mean Dice (nerve root): 0.5457
Saved Dice results to ./ckpt3d_mt\dice_nerve_cropped.txt

python train_3d_multi_augument.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-4 --nerve_root_label 1 --dura_label 2 --save_name "multitask_crop_aug_lr1e4"
=== Case-wise Dice (nerve root only) ===
case003.nii: 0.7545
case006.nii: 0.6193
case010.nii: 0.0284
case011.nii: 0.6945
case036.nii: 0.3204
case037.nii: 0.5543
case039.nii: 0.6781
case040.nii: 0.1894
Mean Dice (nerve root): 0.4799
Saved Dice results to ./ckpt3d_mt\dice_nerve_cropped.txt

キュービック補間
python train_3d_cubic.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cubic-3"
=== Case-wise Dice (nerve root only) ===
case003.nii: 0.7456
case006.nii: 0.6441
case010.nii: 0.0038
case011.nii: 0.7171
case036.nii: 0.5808
case037.nii: 0.5682
case039.nii: 0.6969
case040.nii: 0.3511
Mean Dice (nerve root): 0.5385

python train_3d_cubic.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-4 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cubic-4"
=== Case-wise Dice (nerve root only) ===
case003.nii: 0.7104
case006.nii: 0.5935
case010.nii: 0.1306
case011.nii: 0.6998
case036.nii: 0.5024
case037.nii: 0.6190
case039.nii: 0.7098
case040.nii: 0.3044
Mean Dice (nerve root): 0.5337
Saved Dice results to C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cubic-4\dice_nerve_cubic.txt

python train_3d_cross.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --cv --num_folds 5 --epochs 1000 --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cross-lr1e3"

↑ まちがえてこれを 2 回実行して.pth が多分上書きされてる、.txt は全体のは多分大丈夫。fold0 のみ上書きされてると思う

python train_3d_cross.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --cv --num_folds 5 --epochs 1000 --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cross-lr1e3-2"

だからこれにやりなおした

python test_cubic.py --dataset_root Dataset --checkpoint ./ckpt3d-cubic-3/best_3dunet_nerve.pth --out_dir ./test3d_cubic_out --threshold 0.5

> >

python test_multi.py --dataset_root Dataset --checkpoint ./ckpt3d_mt/multitask_crop_lr1e3.pth --out_dir ./pred3d_mt

> >

    python test_3d.py  --dataset_root Dataset --checkpoint ckpt3d/best_3dunet_nerve.pth  --out_dir pred3d

> >
