import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"GPU Device Count: {torch.cuda.device_count()}")
print(
    f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
)
