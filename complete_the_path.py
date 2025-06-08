import torch, os
import torch.nn as nn
import matplotlib.pyplot as plt
from GI_NN_Mod3 import GI_NN
from utils import IMUDataset, IMUDataset_M2M, IMUDataset_M2M_V2, rotate_preds
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error    # ← for MSE/RMSE
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = "data/dataset_train.txt"
val_path = "data/dataset_test_spiral.txt"

SEQ_LEN = 12
INPUT_SIZE = 7
ANCHOR = 5

START_POINT = 0.25  #0.6
END_POINT = 0.37  #0.75

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
model.load_state_dict(torch.load("chkpts/20250528_161505_good/model_20250528_161505_449.pth"))
# model.load_state_dict(torch.load("chkpts/20241105_165051/model_20241105_165051_61.pth"))
model.to(DEVICE)
model = model.cuda().float()
model.eval()

print("Epoch Done")
running_vloss = 0.0
labels, preds, labels_actual = list(), list(), list()
preds.append([0,0])
labels.append([0,0])
labels_actual.append([0,0])

scale_factor = 0.5

pred_idx = 0
with torch.no_grad():
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
                        vycpu[i][0] + labels[-1][0],
                        vycpu[i][1] + labels[-1][1]
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
                        labels_actual.append([
                        vycpu[i][0][0] + labels_actual[-1][0],
                        vycpu[i][0][1] + labels_actual[-1][1]
                                ])
                    else:
                        if pred_idx % 1 == 0:
                            vy_ = model(vX)
                            # print("Predicted:", vy_)
                            # print("Label:", vy)
                            if len(vy_.shape) < 2 or len(vy.shape) < 2:
                                vy_ = torch.unsqueeze(vy_, dim=0)
                                vy = torch.unsqueeze(vy, dim=0)
                            # if ANCHOR is not None:
                            #     vy = vy[:,-1,:].squeeze(dim=1)
                            vy_cpu = vy_.cpu().tolist()
                            # vy_cpu.reverse()dx
                            vy_cpu[i] = [[scale_factor * dx, scale_factor * dy * 0.5] for dx, dy in vy_cpu[i]]
                            vy_cpu = rotate_preds(vy_cpu)
                            # print("Prediction", len(preds))
                            # print("Labels", len(labels), end="\n\n")
                            preds.append([
                                labels[-1][0] + vy_cpu[i][0][0],
                                # preds[-1][0] - vy_cpu[i][-1][0],
                                labels[-1][1] - vy_cpu[i][0][1]
                                # preds[-1][1] - vy_cpu[i][-1][1]
                                        ])
                            labels.append([
                                labels[-1][0] + vy_cpu[i][0][0],
                                # preds[-batch_idx-offset][0] - vy_cpu[i][-1][0],
                                labels[-1][1] - vy_cpu[i][0][1]
                                # preds[-1][1] - vy_cpu[i][-1][1]
                                        ])
                        labels_actual.append([
                        vycpu[i][0][0] + labels_actual[-1][0],
                        vycpu[i][0][1] + labels_actual[-1][1]
                                ])
                        batch_idx += 1
                        pred_idx += 1
                        if batch_idx > ANCHOR:
                            batch_idx = 1
                            offset += ANCHOR-1

        # except:
            # print("[INFO] Not enough data, proceeding...")
            # break

preds = scaler_val_train.inverse_transform(preds).tolist()
labels = scaler_val_train.inverse_transform(labels).tolist()
labels_actual = scaler_val_train.inverse_transform(labels_actual).tolist()

preds_arr = np.array(preds[1:])  
start_idx = int(START_POINT * len(labels_actual))
end_idx   = int(END_POINT * len(preds_arr))
labs_arr = np.array(labels_actual[start_idx : start_idx + end_idx])
print(labs_arr.shape)

lx, ly = [], []
for idx, l in enumerate(labels):
    lx.append(l[0])
    ly.append(l[1])

lax, lay = [], []
for idx, l in enumerate(labels_actual):
    lax.append(l[0])
    lay.append(l[1])
    

px, py = [], []
for idx, p in enumerate(preds):
    px.append(p[0])
    py.append(p[1])


jump = int(np.ceil(len(px)/len(lax[start_idx:start_idx + end_idx])))

lax[:start_idx].extend(lax[start_idx + end_idx:])
lay[:start_idx].extend(lay[start_idx + end_idx:])

# px, py = px[1:labs_arr.shape[0]], py[1:labs_arr.shape[0]]
# preds_arr = np.array(list(zip(px, py)))

plt.scatter(lax, lay)
plt.scatter(px[1::jump], py[1::jump])
plt.scatter(lax[start_idx:start_idx + end_idx], lay[start_idx:start_idx + end_idx])
plt.title("Trajectory Comparison")
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.legend(["Dual EKF", "GRU Model Prediction", "GT (Dual EKF)"])
plt.show()


# convert to numpy arrays, skipping the dummy [0,0] entry

# per‐step Euclidean distances
distance, path = fastdtw(preds_arr, labs_arr, dist=euclidean)

print(f"DTW distance between predicted and actual trajectories: {distance:.4f}")

# Optionally, you can compute average DTW distance per aligned pair:
avg_dtw_dist = distance / len(path)
print(f"Average DTW distance per aligned pair: {avg_dtw_dist:.4f}")
