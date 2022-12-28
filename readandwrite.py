import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

def dataPrep():
    data = pd.read_csv(Path('data.csv'), index_col= "row_id")
    data = data.to_numpy()
    nullArray = np.zeros(data.shape)
    nullArray = np.isnan(data)
    data[nullArray] = 0
    nullArray = nullArray.astype(int)
    mask_drop = torch.FloatTensor(data.shape[0], data.shape[1]).uniform_() > 0.03
    data = torch.from_numpy(data) * mask_drop 
    return data, torch.from_numpy(nullArray), mask_drop