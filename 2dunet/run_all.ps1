# # python train_3d_aug_crop.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-aug-3-crop"

# # python train_3d_aug_crop.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-4 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-aug-4-crop"

# python train_3d_multi_augument.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --nerve_root_label 1 --dura_label 2 --save_name "multitask_crop_aug_lr1e3"

# python train_3d_multi_augument.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-4 --nerve_root_label 1 --dura_label 2 --save_name "multitask_crop_aug_lr1e4"

# python train_3d_cubic.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cubic-3"

# python train_3d_cubic.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --epochs 1000 --batch_size 1 --lr 1e-4 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cubic-4"

# python train_3d_cross.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset"  --cv  --num_folds 5  --epochs 1000  --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cross-lr1e3"

# python crop_cv.py  --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset"  --cv   --num_folds 5  --epochs 1000  --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cross-crop-lr1e3--"

# python train_3d_cross.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset"  --cv  --num_folds 5  --epochs 1000  --batch_size 1 --lr 1e-3 --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cross-lr1e3-2"

# python multi_cv.py --dataset_root "C:\Users\orilab\Desktop\masumoto\2dunet\Dataset" --cv --num_folds 5 --epochs 1000  --lr 1e-3  --out_dir "C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-cross-multi-lr1e3"

# python multi_cubic_train.py --dataset_root Dataset --out_dir ./ckpt3d_mt_iso_aug --save_name multitask_iso_aug_lr1e3_aug --target_spacing_mm 1.0 1.0 1.0 --augment　--lr 1e-3
# python multi_cubic_train.py --dataset_root Dataset --out_dir ./ckpt3d_mt_iso --save_name multitask_iso_aug_lr1e3 --target_spacing_mm 1.0 1.0 1.0 --lr 1e-3

python train_2D_multi.py  --dataset_root Dataset   --out_dir ./ckpt2d_mt  --crop_x 50 200  --crop_y 45 210

python train_2D.py  --dataset_root Dataset  --out_dir ./ckpt2d_root  --crop_x 50 200  --crop_y 45 210