.\.nnenv\Scripts\activate

https://chatgpt.com/share/69201e2e-7a68-8001-9ec5-a49f70975d52

version の確認コード
python --version
python -c "import torch; print(torch.**version**); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
pip show nnunetv2 | Select-String "Version"
nvidia-smi

fold class1_mean_Dice (nerve root)
0 0.7052576267269284
1 0.7136297393855996
2 0.701002830097064
3 0.7014612666232021
4 0.6977695493790251

GPU は NVIDIA GeForce RTX 4080

VRAM（専用 GPU メモリ）は 16376 MiB ≒ 16 GB

いまの GPU メモリ使用量は 1023 MiB / 16376 MiB（ほぼ空いてる）

Driver 560.94 / CUDA 12.6 と表示（これはドライバが対応している CUDA の目安で、PyTorch 側の CUDA とは別概念）
↓
ResEnc M を使用する


Wed Jan 21 11:42:28 2026       
+-----------------------------------------------------------------------------------------+       
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |       
|-----------------------------------------+------------------------+----------------------+       
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |       
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |       
|                                         |                        |               MIG M. |       
|=========================================+========================+======================|       
|   0  NVIDIA GeForce RTX 4080      WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   33C    P8              6W /  320W |    1023MiB /  16376MiB |      0%      Default |       
|                                         |                        |                  N/A |       
+-----------------------------------------+------------------------+----------------------+       

+-----------------------------------------------------------------------------------------+       
| Processes:                                                                              |       
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |       
|        ID   ID                                                               Usage      |       
|=========================================================================================|       
