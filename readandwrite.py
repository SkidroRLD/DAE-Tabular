import numpy as np
import pandas as pd
from pathlib import Path
import torch

def createMask(shape, prob):
    
    mask_drop = torch.FloatTensor(shape[0], shape[1]).uniform_() > prob
    mask_drop = torch.from_numpy(mask_drop.numpy().astype(float))
    
    return mask_drop

def nullSeparation(data):

    nullArray = np.isnan(data)

    changeData = data.copy()
    changeData[nullArray] = 0

    nullArray = nullArray.astype(int)

    return changeData, nullArray

def dataPrep():
    data = pd.read_csv(Path('data.csv'), index_col= "row_id")
    data = data.to_numpy()
    # nullArray = np.zeros(data.shape)

    data, nullArray = nullSeparation(data)
    
    mask_drop = createMask(data.shape, 0.03)
    data = torch.from_numpy(data) * mask_drop

    return data, torch.from_numpy(nullArray), mask_drop