import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from GI_NN_Mod1 import GI_NN
from torch.utils.data import DataLoader
from utils import IMUDataset, DirectionalGPSLoss, GPSLoss, RecentAndFinalLoss

import wandb, datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque

SEQ_LEN = 24
INPUT_SIZE = 7
LAT_CENTER, LON_CENTER = 388731.70, 3974424.49
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_VAL = 1024
ANCHOR = 24    # usually 10

val_path = "data/dataset_test_circle.txt"

def extract_txt(file_path):
    points = []
    X, y = [], []

    with open(file_path, 'r') as f:
        line = f.readlines()
        f.close()
    for l in line:
        pt = list(map(np.float64, l.split(",")))
        points.append(pt)
    for pt in points:
        X.append(pt[2:])
        y.append(pt[:2])

    return X, y

def calculate_difference(angle1, angle2):
        return np.arctan2(np.sin(angle1 - angle2),
                           np.cos(angle1 - angle2))

def make_vectors(distances):
    vectors = [[0, 0]]
    for i in range(1, distances.size(1)-1):
        # x = distances[0,i,0] - distances[0,i-1,0]
        # y = distances[0,i,1] - distances[0,i-1,0]
        vectors.insert(0, [max(distances[0,:,0])/distances[0,i,0], max(distances[0,:,1])/distances[0,i,1]])
        # vectors.append([x, y])

    vectors.reverse()
    diff = calculate_difference(np.arctan2(vectors[-1][1], vectors[-1][0]),
                                np.arctan2(vectors[0][1], vectors[0][0]))
    print("Angle2: ", np.arctan2(vectors[-1][1], vectors[-1][0]) * (180/np.pi))
    print("Angle1: ", np.arctan2(vectors[0][1], vectors[0][0]) * (180/np.pi))
    print("Difference: ", diff * (180/np.pi))
    positions = vectors.copy()
    for i in range(len(vectors)-1):
        positions[i+1][0] += vectors[i][0]
        positions[i+1][1] += vectors[i][1]

    return np.array(vectors), np.array(positions)

Xv, yv = extract_txt(val_path)  # Change this to test different trajectories
scaler_val = MinMaxScaler((-1, 1))
yv = scaler_val.fit_transform(yv)
scaler_Xval = MinMaxScaler((-1, 1))
Xv = scaler_Xval.fit_transform(Xv)

if __name__ == '__main__':
    dataset = DataLoader(IMUDataset(Xv, yv, seq_len=SEQ_LEN, anchors=ANCHOR), batch_size=1, shuffle=True)
    for i, data in enumerate(dataset):
        X, y = data
        # print(y[0,:,:])
        plt.plot(np.arctan(X[0, 0, :]/X[0, 1, :]), 'r')
        plt.plot(X[0, 2, :], 'b')
        plt.plot(X[0, 4, :], 'c')
        # print(y.shape)
        # print(y[0, 0, :])
        plt.legend(['$\\theta$', '$v_x$', '$a_x$'])
        plt.show()
        pred_vectors, positions = make_vectors(y)
        # print("Predicted Vectors:")
        # print(pred_vectors)
        for i in range(1, pred_vectors.shape[0]):
            plt.quiver(positions[i-1, 0], positions[i-1, 1], 
                       pred_vectors[i, 0], pred_vectors[i, 1],
                       angles='xy', scale_units='xy', scale=1, pivot='tail')
        xmin, xmax = min(positions[:,0]) * 2.5, max(positions[:,0]) * 2.5
        ymin, ymax = min(positions[:,1]) * 2.5, max(positions[:,1]) * 2.5
        plt.xlim(xmin - (xmax-xmin) * 0.08, xmax + (xmax-xmin) * 0.08)
        plt.ylim(ymin - (ymax-ymin) * 0.08, ymax + (ymax-ymin) * 0.08)
        plt.show()

        while(True):
            try:
                plt.show()
            except KeyboardInterrupt:
                plt.close()
                break
        if i == 0:
            break
