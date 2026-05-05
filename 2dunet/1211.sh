python train_3d_crop.py --dataset_root C:\Users\orilab\Desktop\masumoto\2dunet\Dataset --epochs 1000  --batch_size 1  --lr 1e-3 --out_dir C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-3-crop
python train_3d_crop.py --dataset_root C:\Users\orilab\Desktop\masumoto\2dunet\Dataset --epochs 1000  --batch_size 1  --lr 1e-4 --out_dir C:\Users\orilab\Desktop\masumoto\2dunet\ckpt3d-4-crop
python train_3d_multi.py  --dataset_root C:\Users\orilab\Desktop\masumoto\2dunet\Dataset --epochs 1000 --batch_size 1 --lr 1e-3 --nerve_root_label 1 --dura_label 2 --save_name multitask_crop_lr1e3
python train_3d_multi.py  --dataset_root C:\Users\orilab\Desktop\masumoto\2dunet\Dataset --epochs 1000 --batch_size 1 --lr 1e-4 --nerve_root_label 1 --dura_label 2 --save_name multitask_crop_lr1e4

