import torch
import torch.nn as nn
import numpy as np
from readandwrite import dataPrep

device = ("cuda" if torch.cuda.is_available() else "cpu")

class DAE(nn.Module):
    def __init__(self) -> None:
        super(DAE, self).__init__()



dataset, nullArray, mask_drop = dataPrep()
batch = 256
#90 10 train valid split
train_size = int(0.9 * dataset.shape[0])
train_set, valid_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
print(train_set.shape)
# train_loader = 