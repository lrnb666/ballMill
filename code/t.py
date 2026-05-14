import torch
print(torch.__version__)
print(torch.cuda.is_available())

from mamba_ssm import Mamba

print("mamba ok")