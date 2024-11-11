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
    def __init__(self, X, y, seq_len, anchors=None, scaler=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.anchors = anchors
        if scaler is not None:
            self.scaler_x, self.scaler_y = scaler[0], scaler[1]

    def __len__(self):
        return (len(self.X) - (self.seq_len + 1))

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float).T
        Y = torch.tensor(self.y[idx:idx+self.seq_len], dtype=torch.float).T
        y = torch.tensor([
            self.y[idx + self.seq_len + 1][0] - self.y[idx][0],
            self.y[idx + self.seq_len + 1][1] - self.y[idx][1]
                         ], dtype=torch.float)
        X = torch.cat([X, Y], dim=0)
        if self.anchors is not None:
            y_multi = torch.zeros((self.anchors, 2))
            for i in reversed(range(self.anchors)):
                y_multi[i,:] = torch.tensor([
                            self.y[idx + self.seq_len - i][0] - self.y[idx][0],
                            self.y[idx + self.seq_len - i][1] - self.y[idx][1]
                            ], dtype=torch.float)
            
            return (X, y_multi)       # For ANCHORS number of time-step loss
        
        else:
            return (X, y)             # For single (last) time-step loss
        
class GPSLoss(nn.Module):
    def __init__(self, x_bias = 0.5, y_bias = 0.5):
        super(GPSLoss, self).__init__()
        self.x_bias = x_bias
        self.y_bias = y_bias
        self.huber = nn.HuberLoss()

    def forward(self, y_pred, y_true):
        x_loss = self.huber(y_pred[:, 0], y_true[:, 0])
        y_loss = self.huber(y_pred[:, 1], y_true[:, 1])

        return x_loss*self.x_bias + y_loss*self.y_bias
    
class RecentAndFinalLoss(nn.Module):
    def __init__(self, anchors, recent_weight=0.4, final_weight=0.4, dir_weight=0.2):
        super(RecentAndFinalLoss, self).__init__()
        self.recent_weight = recent_weight
        self.final_weight = final_weight
        self.dir_weight = dir_weight
        self.epsilon = 1e-8
        self.anchors = anchors
        self.loss_fn = nn.HuberLoss()  # or nn.MSELoss() or another suitable loss
        self.cosine_loss = nn.CosineSimilarity(dim=-1, eps=self.epsilon)

    def forward(self, predictions, targets):
        if self.anchors is None or len(predictions.size()) < 3:
            # Using arctan2
            pred_angle = torch.atan2(predictions[..., 1], predictions[..., 0] + self.epsilon)
            target_angle = torch.atan2(targets[..., 1], targets[..., 0] + self.epsilon)
            directional_loss = torch.mean(torch.abs(pred_angle - target_angle))

            # Using cosine similarity
            directional_loss = 1 - self.cosine_loss(predictions, targets).mean()
            
            return self.loss_fn(predictions, targets) * (1-self.final_weight) + directional_loss * self.final_weight
            return self.loss_fn(predictions, targets)
        
        recent_predictions = predictions[:, -1*self.anchors:-1, :]
        recent_targets = targets[:, -1*self.anchors:-1, :]
        recent_loss = self.loss_fn(recent_predictions, recent_targets)

        # Using arctan2
        # pred_slope = torch.atan2(recent_predictions[..., 1], recent_predictions[..., 0] + self.epsilon)
        # target_slope = torch.atan2(recent_targets[..., 1], recent_targets[..., 0] + self.epsilon)
        # sin_loss = torch.sin(pred_slope) - torch.sin(target_slope)
        # cos_loss = torch.cos(pred_slope) - torch.cos(target_slope)
        # directional_loss = torch.mean((sin_loss)**2 + (cos_loss)**2)


        # Using cosine similarity
        directional_loss = 1 - self.cosine_loss(recent_predictions, recent_targets).mean()

        final_prediction = predictions[:, -1, :]
        final_target = targets[:, -1, :]
        final_loss = self.loss_fn(final_prediction, final_target)
        
        combined_loss = (self.recent_weight * recent_loss) + (self.final_weight * final_loss) + (directional_loss * self.dir_weight)
        # combined_loss = (self.recent_weight * recent_loss) + (self.final_weight * final_loss)
        return combined_loss
    
class DirectionalGPSLoss(nn.Module):
    def __init__(self, alpha = 0.5, beta = 0.5):
        super(DirectionalGPSLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.huber = nn.HuberLoss()

    def forward(self, pred, target):
        if torch.isnan(pred).any():
            print("NaN detected in pred or target!")
        if torch.isinf(pred).any() or torch.isinf(target).any():
            print("Infinity detected in pred or target!")
        if pred.abs().max() > 1e6 or target.abs().max() > 1e6:
            print("Large values detected in pred or target!")
        pred_angle = torch.atan2(pred[..., 1], pred[..., 0])
        target_angle = torch.atan2(target[..., 1], target[..., 0])
        angle_diff = torch.abs(pred_angle - target_angle)
        
        angle_diff = torch.remainder(angle_diff, 2 * torch.pi)
        angle_diff = torch.where(angle_diff > torch.pi, 2 * torch.pi - angle_diff, angle_diff)
        angle_diff = angle_diff.mean()
        
        error_loss = self.huber(pred, target)

        total_loss = self.alpha * error_loss + self.beta * angle_diff
        return total_loss

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
    
    ANCHOR = 7

    dataset = IMUDataset(X, y, seq_len=21, anchors=ANCHOR, scaler=y[0])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    for i, data in enumerate(dataloader):
        X, y = data
        print(X.shape, y.shape)
        break
    # pt = next(iter(dataset))
    # print(pt[0].shape)