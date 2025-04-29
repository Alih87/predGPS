import torch, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from copy import copy
import datetime
from collections import deque
import numpy as np

def collate_fn(batch):
    X, y = zip(*batch)
    return torch.cat(X), torch.cat(y)

def calculate_difference(angle1, angle2):
        return torch.atan2(torch.sin(angle1 - angle2),
                           torch.cos(angle1 - angle2))

def make_distances(points):
    dists = []
    for idx in range(len(points[:-1])):
        dists.append(np.linalg.norm(points[idx+1] - points[idx]))
    dists.append(0)
    return dists

def make_vectors(distances, routine=False):
    vectors = []
    positions = []
    for i in range(1, distances.size(1)-1):
        # x = distances[0,i,0] - distances[0,i-1,0]
        # y = distances[0,i,1] - distances[0,i-1,0]
        # vectors.append([max(distances[0,:,0])/distances[0,i,0], max(distances[0,:,1])/distances[0,i,1]])
        vectors.insert(0, [distances[0,i,0], distances[0,i,1]])
        # vectors.append([x, y])
    vectors.append([0, 0])
    vectors.reverse()
    # diff = calculate_difference(np.arctan2(vectors[-1][1], vectors[-1][0]),
    #                             np.arctan2(vectors[0][1], vectors[0][0]))
    # print("Angle2: ", np.arctan2(vectors[-1][1], vectors[-1][0]) * (180/np.pi))
    # print("Angle1: ", np.arctan2(vectors[0][1], vectors[0][0]) * (180/np.pi))
    # print("Difference: ", diff * (180/np.pi))
    if not routine:
        positions = vectors.copy()
        for i in range(len(vectors)-1):
            positions[i+1][0] += vectors[i][0]
            positions[i+1][1] += vectors[i][1]

        return np.array(vectors), np.array(positions)
    
    else:
        return torch.tensor(vectors)

class IMUDataset(Dataset):
    def __init__(self, X, y, seq_len, anchors=None, scaler=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.anchors = anchors
        if scaler is not None:
            self.scaler_x, self.scaler_y = scaler[0], scaler[1]

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float).T
        Y = torch.tensor(self.y[idx:idx + self.seq_len], dtype=torch.float).T

        y = torch.tensor([
            self.y[min(idx + self.seq_len + 1, len(self.y) - 1)][0] - self.y[idx + self.seq_len][0],
            self.y[min(idx + self.seq_len + 1, len(self.y) - 1)][1] - self.y[idx + self.seq_len][1]
        ], dtype=torch.float)

        # X = torch.concat([X, Y], dim=0)

        if self.anchors is not None:
            y_multi = torch.zeros((self.anchors, 2))
            for i in range(self.anchors):
                anchor_idx = min(idx + self.seq_len - self.anchors + i, len(self.y) - 1)
                y_multi[i, :] = torch.tensor([
                    self.y[min(anchor_idx + self.anchors - i + 1, len(self.y) - 1)][0] - self.y[anchor_idx][0],
                    self.y[min(anchor_idx + self.anchors - i + 1, len(self.y) - 1)][1] - self.y[anchor_idx][1]
                ], dtype=torch.float)

            return (X, y_multi)  # Return the sequence and multi-step displacements

        else:
            return (X, y)  # Return the sequence and single final displacement
        
class IMUDataset_M2M(Dataset):
    def __init__(self, X, y, seq_len, anchors=None, scaler=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.anchors = anchors
        if scaler is not None:
            self.scaler_x, self.scaler_y = scaler[0], scaler[1]

    def __len__(self):
        return len(self.X) - self.seq_len - self.anchors

    def __getitem__(self, idx):
        # Input sequence
        X = torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float).T  # [features, seq_len]

        # Compute future displacements relative to the last input point
        base_idx = idx + self.seq_len - 1
        y_multi = torch.zeros((self.anchors, 2))
        base_point = self.y[base_idx]
        for i in range(self.anchors):
            future_idx = min(base_idx + i + 1, len(self.y) - 1)
            y_multi[i, :] = torch.tensor([
                self.y[future_idx][0] - base_point[0],
                self.y[future_idx][1] - base_point[1]
            ], dtype=torch.float)

        return X, y_multi  # many-to-many displacements
    
class IMUDataset_M2M_V2(Dataset):
    def __init__(self, X, y, seq_len, anchors=None, scaler=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.anchors = anchors
        if scaler is not None:
            self.scaler_x, self.scaler_y = scaler[0], scaler[1]

    def __len__(self):
        return len(self.X) - self.seq_len - self.anchors

    def __getitem__(self, idx):
        # Input sequence
        X = torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float).T  # [features, seq_len]

        # Base point (absolute position) to compute displacements and track absolute future positions
        base_idx = idx + self.seq_len - 1
        base_point = torch.tensor(self.y[base_idx], dtype=torch.float)

        # Initialize displacement and absolute position tensors
        y_multi_disp = torch.zeros((self.anchors, 2))
        y_multi_abs = torch.zeros((self.anchors, 2))

        for i in range(self.anchors):
            future_idx = min(base_idx + i + 1, len(self.y) - 1)
            future_point = torch.tensor(self.y[future_idx], dtype=torch.float)
            y_multi_disp[i, :] = future_point - base_point
            y_multi_abs[i, :] = future_point

        return X, y_multi_disp, y_multi_abs  # Input, displacements, and absolute positions

        
class GPSLoss(nn.Module):
    def __init__(self, x_bias = 1, y_bias = 1):
        super(GPSLoss, self).__init__()
        self.x_bias = x_bias
        self.y_bias = y_bias
        self.huber = nn.HuberLoss()

    def forward(self, y_pred, y_true):
        x_loss = self.huber(y_pred[:, 0], y_true[:, 0])
        y_loss = self.huber(y_pred[:, 1], y_true[:, 1])

        return x_loss*self.x_bias + y_loss*self.y_bias
    
    # def point_spacing(self, gt, pred):

class StepDistanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(StepDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_abs, target_abs):
        # pred_abs: [batch, anchors, 2]
        # target_abs: [batch, anchors, 2]

        pred_steps = pred_abs[:, 1:, :] - pred_abs[:, :-1, :]  # [batch, anchors-1, 2]
        target_steps = target_abs[:, 1:, :] - target_abs[:, :-1, :]  # [batch, anchors-1, 2]

        pred_dists = torch.norm(pred_steps, dim=2)  # [batch, anchors-1]
        target_dists = torch.norm(target_steps, dim=2)  # [batch, anchors-1]

        loss = F.mse_loss(pred_dists, target_dists, reduction=self.reduction)

        return loss
    
class RecentAndFinalLoss(nn.Module):
    def __init__(self, anchors, vec_weights=0.1, recent_weight=0.4, step_weight=0.2, dir_weight=0.3):
        # Best params: recent_weight = 0.45, final_weight = 0.10, dir_weight = o.45
        super(RecentAndFinalLoss, self).__init__()
        self.recent_weight = recent_weight
        self.step_weight = step_weight
        self.dir_weight = dir_weight
        self.vec_weights = vec_weights
        self.epsilon = 1e-8
        self.anchors = anchors

        self.loss_fn = GPSLoss(x_bias=0.5, y_bias=0.5)  # Best params: x_bias=0.35, y_bias=0.65
        self.dir_loss_fn = DirectionalGPSLoss(alpha=0.5, beta=0.5)  # Best params: alpha=0.5, beta=0.5
        self.vector_loss = VectorLoss(self.anchors)
        self.step_loss = StepDistanceLoss()
        
        self.count = 0
        self.FACTOR = 54
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.cosine_loss = nn.CosineSimilarity(dim=-1, eps=self.epsilon)

    def forward(self, predictions, targets):
        if self.anchors is None or len(predictions.size()) < 3:
            directional_loss = (1-self.cosine_loss(predictions, targets)).mean()

            return self.loss_fn(predictions, targets) * (self.final_weight+self.recent_weight) + (directional_loss * self.dir_weight)
            # return self.loss_fn(predictions, targets)
        
        # For recent loss
        recent_predictions = predictions[:, :, :]
        recent_targets = targets[:, :, :]
        
        # For vector loss
        recent_pred_vectors = make_vectors(recent_predictions, routine=True)
        recent_tgt_vectors = make_vectors(recent_targets, routine=True)

        # For step distance loss
        abs_predictions = torch.abs(predictions)
        abs_targets = torch.abs(targets)

        v_loss = self.vector_loss(recent_pred_vectors, recent_tgt_vectors)
        # v_loss = 0
        # print(v_loss)

        if self.count % self.FACTOR == 0:
            recent_pred_vectors, recent_pred_positions = make_vectors(recent_predictions.detach().cpu())
            recent_tgt_vectors, recent_tgt_positions = make_vectors(recent_targets.detach().cpu())

            plt.figure()
            pred_quivers = []
            tgt_quivers = []

            for i in range(1, recent_pred_vectors.shape[0]):
                p = plt.quiver(recent_pred_positions[i-1, 0], recent_pred_positions[i-1, 1], 
                            recent_pred_vectors[i, 0], recent_pred_vectors[i, 1],
                            angles='xy', scale_units='xy', scale=1, pivot='tail', color='b')
                pred_quivers.append(p)

            for i in range(1, recent_tgt_vectors.shape[0]):
                q = plt.quiver(recent_tgt_positions[i-1, 0], recent_tgt_positions[i-1, 1], 
                            recent_tgt_vectors[i, 0], recent_tgt_vectors[i, 1],
                            angles='xy', scale_units='xy', scale=1, pivot='tail', color='r')
                tgt_quivers.append(q)

            plt.legend([pred_quivers[0], tgt_quivers[0]], ['Predicted', 'Target'])

            all_x = np.concatenate([recent_pred_positions[:, 0], recent_tgt_positions[:, 0]])
            all_y = np.concatenate([recent_pred_positions[:, 1], recent_tgt_positions[:, 1]])
            xmin, xmax = all_x.min() * 2.5, all_x.max() * 2.5
            ymin, ymax = all_y.min() * 2.5, all_y.max() * 2.5

            plt.xlim(xmin - (xmax - xmin) * 0.08, xmax + (xmax - xmin) * 0.08)
            plt.ylim(ymin - (ymax - ymin) * 0.08, ymax + (ymax - ymin) * 0.08)
            os.makedirs(f"vectors/{self.time_stamp}", exist_ok=True)
            plt.savefig(f"vectors/{self.time_stamp}/vectors_epoch_{self.count//self.FACTOR}.png")
            plt.clf()
            plt.close()
            
        self.count += 1
        # pred_vectors = recent_predictions[:, 1:, :] - recent_predictions[:, :-1, :]
        # target_vectors = recent_targets[:, 1:, :] - recent_targets[:, :-1, :]

        recent_loss = self.loss_fn(recent_predictions, recent_targets)
        dir_loss = self.dir_loss_fn(recent_predictions, recent_targets)
        step_loss =  self.step_loss(abs_predictions, abs_targets)

        ################ No longer using final loss ############
        # final_prediction = predictions[:, -1, :]
        # final_target = targets[:, -1, :]
        # final_loss = self.loss_fn(final_prediction, final_target)
        
        total_weights = v_loss + recent_loss + step_loss
        combined_loss = ((self.vec_weights / total_weights) * v_loss)
        + ((self.recent_weight / total_weights) * recent_loss) 
        + ((self.step_weight / total_weights) * step_loss)
        + ((dir_loss / total_weights) * self.dir_weight)
        # combined_loss = (self.recent_weight * recent_loss) + (self.final_weight * final_loss)
        
        return combined_loss

class DirectionalGPSLoss(nn.Module):
    def __init__(self, alpha=0.30, beta=0.70):
        # Best params: alpha = 0.30, beta = 0.70
        # Good params: alpha = 0.20, beta = 0.80
        super(DirectionalGPSLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        if torch.isnan(pred).any():
            print("NaN detected in pred or target!")
        if torch.isinf(pred).any() or torch.isinf(target).any():
            print("Infinity detected in pred or target!")
        if pred.abs().max() > 1e6 or target.abs().max() > 1e6:
            print("Large values detected in pred or target!")

        pred_angles = torch.atan2(pred[...,1], pred[...,0])
        target_angles = torch.atan2(target[...,1], target[...,0])

        angle_diff = self.calculate_difference(target_angles, pred_angles)
        angle_diff = torch.remainder(angle_diff, 2 * torch.pi)
        angle_diff = torch.where(angle_diff > torch.pi, 2*torch.pi - angle_diff, angle_diff)
        angle_diff = angle_diff.mean()
        
        pred_angle = torch.atan2(pred[:, -1, 1], pred[:, -1, 0])
        target_angle = torch.atan2(target[:, -1, 1], target[:, -1, 0])

        single_diff = self.calculate_difference(target_angle, pred_angle)
        single_diff = torch.remainder(single_diff, 2 * torch.pi)
        single_diff = torch.where(single_diff > torch.pi, 2*torch.pi - single_diff, single_diff)
        single_diff = single_diff.mean()

        total_loss = self.alpha * angle_diff + self.beta * single_diff

        return total_loss
        
    def calculate_difference(self, angle1, angle2):
        return torch.atan2(torch.sin(angle1 - angle2),
                           torch.cos(angle1 - angle2))
    
    def get_angle_diff(self, distances):
        vectors = [[0, 0]]
        for i in range(1, distances.size(1)-1):
            vectors.insert(0, [max(distances[0,:,0]) / distances[0,i,0],
                               max(distances[0,:,1]) / distances[0,i,1]])
        vectors = torch.tensor(vectors)
        diff = abs(self.calculate_difference(torch.atan2(vectors[0, 1], vectors[0, 0]),
                                         torch.atan2(vectors[-1, 1], vectors[-1, 0])))

        return diff
    
class VectorLoss(nn.Module):
    def __init__(self, anchors, mag_w=0.40, dir_w=0.60):
        super(VectorLoss, self).__init__()
        self.mag_w = mag_w
        self.dir_w = dir_w

    def forward(self, pred, tgt):
        pred_n = self.normalize(pred)
        tgt_n = self.normalize(tgt)

        # Focus only on second vector (your setup)
        dot_ = torch.sum(tgt_n[1] * pred_n[1])
        dot_ = torch.clamp(dot_, -1.0, 1.0)

        # Force dot product toward +1
        direction_loss = 1 - dot_

        # Magnitude difference
        mag_diffs = torch.abs(torch.norm(pred, dim=1) - torch.norm(tgt, dim=1))

        # Combine
        combined = self.mag_w * mag_diffs.mean() + self.dir_w * direction_loss
        combined = torch.nan_to_num(combined, nan=0.0)

        return combined

    def normalize(self, vectors):
        assert isinstance(vectors, torch.Tensor), "Input must be a PyTorch tensor"
        norms = vectors.norm(p=2, dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        return vectors / norms

class VectorLoss_v2(nn.Module):
    def __init__(self, mag_w=0.5, dir_w=0.5):
        super(VectorLoss, self).__init__()
        # Register priority mask as a buffer so it moves with the model (to GPU/CPU)
        # self.register_buffer('priority_mask', torch.linspace(0, 1, anchors - 1))
        self.mag_w = mag_w
        self.dir_w = dir_w

    def forward(self, pred, tgt):
        pred_n = self.normalize(pred)
        tgt_n = self.normalize(tgt)

        # Compute angle between vectors
        dot_ = torch.sum(tgt_n[1] * pred_n[1])
        dot_ = torch.clamp(dot_, -1.0, 1.0)
        angle_diff = torch.acos(dot_)

        # Compute magnitude difference
        mag_diffs = torch.abs(torch.norm(pred, dim=1) - torch.norm(tgt, dim=1))

        # Combine angle and magnitude penalties
        # combined = self.mag_w * mag_diffs + self.dir_w * angle_diff
        combined = angle_diff
        combined = torch.nan_to_num(combined, nan=0.0)

        # Apply priority mask (broadcasted)
        return torch.sum(combined)

    def normalize(self, vectors):
        assert isinstance(vectors, torch.Tensor), "Input must be a PyTorch tensor"
        norms = vectors.norm(p=2, dim=1, keepdim=True)  # Compatible with older PyTorch
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)  # Avoid division by zero
        return vectors / norms

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
    
    ANCHOR = 32

    dataset = IMUDataset(X, y, seq_len=64, anchors=ANCHOR, scaler=y[0])
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    for i, data in enumerate(dataloader):
        X, y = data
        break
    # pt = next(iter(dataset))
    # print(pt[0].shape)