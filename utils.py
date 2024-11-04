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
        if scaler is not None:
            self.scaler_x, self.scaler_y = scaler[0], scaler[1]

    def __len__(self):
        return (len(self.X) - (self.seq_len + 1))

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float).T
        Y = torch.tensor(self.y[idx:idx+self.seq_len], dtype=torch.float).T
        y = torch.tensor([
            self.y[idx + self.seq_len + 1][0] - self.y[idx][0],
            self.y[idx + self.seq_len + 1][1] - self.y[idx][1],
            self.y[idx + self.seq_len + 1][2] - self.y[idx][2]
                         ], dtype=torch.float)
        X = torch.cat([X, Y[:-1]], dim=0)
        
        return (X, y)
        # return (torch.unsqueeze(torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float).T, dim=0), torch.tensor(np.sum(self.y[idx:idx+self.seq_len], axis=0), dtype=torch.float64))

class GPSLoss(nn.Module):
    def __init__(self, x_bias = 1, y_bias = 1, yaw_bias = 1):
        super(GPSLoss, self).__init__()
        self.x_bias = x_bias
        self.y_bias = y_bias
        self.yaw_bias = yaw_bias

    def forward(self, y_pred, y_true):
        x_loss = torch.mean(torch.abs(y_pred[:, 0] - y_true[:, 0]))
        y_loss = torch.mean(torch.abs(y_pred[:, 1] - y_true[:, 1]))
        yaw_loss = torch.mean(torch.abs(y_pred[:, 2] - y_true[:, 2]))

        return x_loss*self.x_bias + y_loss*self.y_bias + yaw_loss*self.yaw_bias

if __name__ == '__main__':
    points = []
    X, y = [], []
    
    file_path = "data/data_fallback.txt"
    with open(file_path, 'r') as f:
        line = f.readlines()
        f.close()
    for l in line:
        pt = list(map(np.float64, l.split(",")))
        points.append(pt)
    for pt in points:
        X.append(pt[2:])
        y.append(pt[:2]+[pt[4]])

    dataset = IMUDataset(X, y, seq_len=7, scaler=y[0])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    for i, data in enumerate(dataloader):
        X, y = data
        print(X.shape, y.shape)
    # pt = next(iter(dataset))
    # print(pt[0].shape)