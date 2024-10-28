import torch, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from copy import copy
from collections import deque
import numpy as np

def collate_fn(batch):
    X, y = zip(*batch)
    return torch.cat(X), torch.cat(y)

class IMUDataset(Dataset):
    def __init__(self, X, y, seq_len, scaler=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.scaler_x, self.scaler_y = scaler[0], scaler[1]

    def __len__(self):
        return (len(self.X) - (self.seq_len + 1))

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float).T
        Y = torch.tensor(self.y[idx:idx+self.seq_len], dtype=torch.float64).T
        for i in range(Y.shape[1]):
            Y[:,i][0] = Y[:,i][0] - self.scaler_x
            Y[:,i][1] = Y[:,i][1] - self.scaler_y
        X = torch.cat([X, Y], dim=0)
        return (X, Y[:,-1])
        # return (torch.unsqueeze(torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float).T, dim=0), torch.tensor(np.sum(self.y[idx:idx+self.seq_len], axis=0), dtype=torch.float64))

if __name__ == '__main__':
    points = []
    X, y = [], []
    
    file_path = "data/data.txt"
    with open(file_path, 'r') as f:
        line = f.readlines()
        f.close()
    for l in line:
        pt = list(map(np.float64, l.split(",")))
        points.append(pt)
    for pt in points:
        X.append(pt[2:])
        y.append(pt[:2])

    dataset = IMUDataset(X, y, seq_len=7, scaler=y[0])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    for i, data in enumerate(dataloader):
        X, y = data
        print(X.shape, y.shape)
        break
    # pt = next(iter(dataset))
    # print(pt[0].shape)