import numpy as np
import torch

a = torch.zeros((5,5))
b = torch.ones((5,2))
c = torch.cat([a,b],dim = 1)
print(c.shape)