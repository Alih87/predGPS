import torch, os
import torch.nn as nn
import matplotlib.pyplot as plt
from GI_NN_Mod2 import GI_NN
from utils import IMUDataset, IMUDataset_M2M, IMUDataset_M2M_V2, trajectory_construct_M2M, rotate_preds
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = "data/dataset_train.txt"
val_path = "data/dataset_test_spiral.txt"

SEQ_LEN = 64
INPUT_SIZE = 7
ANCHOR = 32

START_POINT = 0.2  #0.6
END_POINT = 0.3    #0.75

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def normalize_data(X):
    x = np.asanyarray(X)
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

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

Xv, yv = extract_txt(val_path)  # Change this to test different trajectories
Xt, yt = extract_txt(train_path)

scaler_val = MinMaxScaler((-1,1))
yv = scaler_val.fit_transform(yv)
scaler_Xval = MinMaxScaler((-1,1))
Xv = scaler_Xval.fit_transform(Xv)

scaler_val_train = MinMaxScaler((-1,1))
yt = scaler_val_train.fit_transform(yt)

# Xv = np.concatenate((Xv, yv), axis=1)

yv = (np.asanyarray(yv)).tolist()
yt = (np.asanyarray(yt)).tolist()

validation_loader = DataLoader(IMUDataset_M2M(Xv, yv, seq_len=SEQ_LEN, anchors=ANCHOR), batch_size=1, shuffle=False)

model = GI_NN(input_size=INPUT_SIZE, output_channels=2, anchors=ANCHOR, SEQ_LEN=SEQ_LEN)
model.load_state_dict(torch.load("chkpts/20250508_065917/model_20250508_065917_249.pth"))
# model.load_state_dict(torch.load("chkpts/20241105_165051/model_20241105_165051_61.pth"))
model.to(DEVICE)
model = model.cuda().float()
model.eval()

print("Epoch Done")
running_vloss = 0.0
labels, preds = list(), list()
preds.append([0,0])
labels.append([0,0])
only_preds = list()

with torch.no_grad():
    FIRST_PRED = True
    for ii, vdata in enumerate(validation_loader):
        # try:-
            vX, vy = vdata
            vX = vX.cuda().float()
            vy = vy.cuda().float()
            vX.to(DEVICE)
            vy.to(DEVICE)
            offset = 0
            batch_idx = 1
            vycpu = vy.cpu().tolist()
            for i in range(vy.shape[0]):
                if ANCHOR is None:
                    if ii < int(START_POINT * validation_loader.__len__()) or ii > int(END_POINT * validation_loader.__len__()):
                        labels.append([
                        vycpu[i][0] + labels[0][0],
                        vycpu[i][1] + labels[0][1]
                                ])
                    else:
                        vy_ = model(vX)
                        if len(vy_.shape) < 2 or len(vy.shape) < 2:
                            vy_ = torch.unsqueeze(vy_, dim=0)
                            vy = torch.unsqueeze(vy, dim=0)
                        if ANCHOR is not None:
                            vy = vy[:,-1,:].squeeze(dim=1)
                        vy_cpu = vy_.cpu().tolist()
                        preds.append([
                            vy_cpu[i][0] + labels[-1][0],
                            # preds[-1][0] - vy_cpu[i][0],
                            vy_cpu[i][1] + labels[-1][1]
                            # preds[-1][1] - vy_cpu[i][1]
                                    ])
                        labels.append([
                            labels[-1][0] + vy_cpu[i][0],
                            # preds[-1][0] - vy_cpu[i][0],
                            labels[-1][1] + vy_cpu[i][1]
                            # preds[-1][1] - vy_cpu[i][1]
                                    ])
                        batch_idx += 1
                else:
                    if ii < int(START_POINT*validation_loader.__len__()) or ii > int(END_POINT*validation_loader.__len__()):
                        labels.append([
                        vycpu[i][0][0] + labels[-1][0],
                        vycpu[i][0][1] + labels[-1][1]
                                ])
                    else:
                        vy_ = model(vX)
                        # print("Predicted:", vy_)
                        # print("Label:", vy)
                        if len(vy_.shape) < 2 or len(vy.shape) < 2:
                            vy_ = torch.unsqueeze(vy_, dim=0)
                            vy = torch.unsqueeze(vy, dim=0)
                        if ANCHOR is not None:
                            vy = vy[:,-1,:].squeeze(dim=1)
                        vy_cpu = vy_.cpu().tolist()
                        vy_cpu = rotate_preds(vy_cpu)
                        if len(labels) < 32:
                            preds.append([
                                labels[-1][0] + vy_cpu[i][0][0],
                                # preds[-1][0] - vy_cpu[i][-1][0],
                                labels[-1][1] + vy_cpu[i][0][1]
                                # preds[-1][1] - vy_cpu[i][-1][1]
                                        ])
                            labels.append([
                                labels[-1][0] + vy_cpu[i][0][0],
                                # preds[-batch_idx-offset][0] - vy_cpu[i][-1][0],
                                labels[-1][1] + vy_cpu[i][0][1]
                                # preds[-1][1] - vy_cpu[i][-1][1]
                                        ])
                            labels.extend(vy_cpu[i][1:])
                            batch_idx += 1
                            if batch_idx > ANCHOR:
                                batch_idx = 1
                                offset += ANCHOR-1
                        else:
                            labels = trajectory_construct_M2M(vy_cpu[i], labels, ANCHOR)
                            if FIRST_PRED:
                                preds = trajectory_construct_M2M(vy_cpu[i], labels[-ANCHOR-1:], ANCHOR)
                                # only_preds = trajectory_construct_M2M(vy_cpu[i], labels[-ANCHOR-1:], ANCHOR)
                                FIRST_PRED = False
                            else:
                                preds = trajectory_construct_M2M(vy_cpu[i], preds, ANCHOR)
                                # only_preds = trajectory_construct_M2M(vy_cpu[i], only_preds, ANCHOR)
                            
                            batch_idx += 1
                            if batch_idx > ANCHOR:
                                batch_idx = 1
                                offset += ANCHOR-1
                    # print(len(labels))
                    # print(len(preds))
        # except:
            # print("[INFO] Not enough data, proceeding...")
            # break

preds = scaler_val.inverse_transform(preds).tolist()
labels = scaler_val.inverse_transform(labels).tolist()

lx, ly = [], []
for idx, l in enumerate(labels):
    lx.append(l[0])
    ly.append(l[1])

px, py = [], []
for idx, p in enumerate(preds):
    px.append(p[0])
    py.append(p[1])

plt.scatter(lx, ly)
plt.scatter(px[1:], py[1:])
# plt.scatter(opx[1:], opy[1:])
plt.title("Trajectory Comparison")
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.legend(["Actual", "Predicted"])
plt.show()
