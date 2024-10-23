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
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.X) - (self.seq_len + 1))
    
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float).T, torch.tensor([self.y[idx+self.seq_len][0], self.y[idx+self.seq_len][1]], dtype=torch.float64))
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

    dataset = IMUDataset(X, y, seq_len=7)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, data in enumerate(dataloader):
        X, y = data
        print(X.shape)
        print(y.shape)
    # pt = next(iter(dataset))
    # print(pt[0].shape)