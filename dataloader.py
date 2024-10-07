import torch, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from copy import copy
from collections import deque
import numpy as np

class IMUDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.points) - (self.seq_len - 1)
    
    def __getitem__(self, idx):
        return (torch.unsqueeze(torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float).T, dim=0), torch.tensor([self.y[idx+self.seq_len][0], self.y[idx+self.seq_len][1]], dtype=torch.float64))
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

    print(y)
    dataset = IMUDataset(X, y, seq_len=7)
    pt = next(iter(dataset))
    print(pt[0].shape)