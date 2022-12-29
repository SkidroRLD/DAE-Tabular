import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from readandwrite import dataPrep
from torch.utils.data import DataLoader

device = ("cuda" if torch.cuda.is_available() else "cpu")

class DAE(nn.Module):
    def __init__(self) -> None:
        super(DAE, self).__init__()
        self.feature_embed = nn.Linear(80, 60)
        self.mask_embed = nn.Linear(80,20)

        self.seq1 = nn.Sequential(
            nn.Linear(80,48),
            nn.LayerNorm(48)
        )
        self.mlp = nn.Sequential(
            nn.Linear(48, 512),
            nn.Mish(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.LayerNorm(256),
            nn.Linear(256, 80),
            nn.Mish(),
            nn.LayerNorm(80)
        )
    
    def forward(self, x, mask):
        x = self.feature_embed(x)
        y = self.mask_embed(mask)
        z = torch.cat([x,y], dim = 1)
        x = self.seq1(z)
        x = self.mlp(x)
        return x


dataset, nullArray, mask_drop = dataPrep()
dataset = dataset.to(device=device)
nullArray = nullArray.to(device=device)
mask_drop = mask_drop.to(device=device)
mask_drop = mask_drop * nullArray
batch = 4000

n_epochs = 15
valid_every = 3

#masked MSE loss borrowed from SebastianVanGerwen
class MaskedMSELoss(nn.Module):
    # Mask should be 1 for masked value, 0 for unmasked value 
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')
    
    def forward(self, inputs, target, mask):
        loss = self.loss(inputs, target)
        return torch.mean(loss * (1 - mask))


#90 10 train valid split
train_size = int(0.9 * dataset.shape[0])
train_set, valid_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch, shuffle=True)

model = DAE().to(device = device)

mseloss = MaskedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)
model.train()

for epoch in tqdm(range(n_epochs)):
    x = 0
    for i in tqdm(enumerate(train_loader)):
        pred = model(i[1],mask_drop[batch*x:batch*(x+1)])
        optimizer.zero_grad()
        loss = mseloss(i[1], pred, mask_drop[batch*x:batch*(x+1)])
        loss.backward()
        optimizer.step()
        x += 1
    if epoch + 1 % valid_every == 0:
        model.eval()
        y = 0
        for j in enumerate(valid_loader):
            pred = model(j,mask_drop[batch*y:batch*(y+1)])
            loss = mseloss(j, pred, mask_drop[batch*y:batch*(y+1)])
            y += 1 


