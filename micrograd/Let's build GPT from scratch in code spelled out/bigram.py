import torch
from torch import nn
import torch.nn.functional as F

# hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# ---------------------------------

print(device)
