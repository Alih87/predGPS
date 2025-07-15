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
from scipy.interpolate import interp1d

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = "data/dataset_train.txt"
val_path = "data/dataset_test_spiral.txt"

SEQ_LEN = 12
INPUT_SIZE = 7
ANCHOR = 5

START_POINT = 0.58 #0.64  #0.6
END_POINT = 0.84 #0.82  #0.75

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

import numpy as np
from scipy.interpolate import interp1d

def arc_length(path):
    diffs = np.diff(path, axis=0)
    seg   = np.hypot(diffs[:,0], diffs[:,1])
    return np.concatenate(([0.0], np.cumsum(seg)))    # s[0]=0

def rmse_vs_distance(pred_xy, gt_xy, ds=0.05):
    s_pred = arc_length(pred_xy)
    s_gt   = arc_length(gt_xy)

    s_max  = min(s_pred[-1], s_gt[-1])
    s_grid = np.arange(0, s_max, ds)

    # interpolate E and N on uniform distance grid
    fE_pred = interp1d(s_pred, pred_xy[:,0], kind="linear")
    fN_pred = interp1d(s_pred, pred_xy[:,1], kind="linear")
    fE_gt   = interp1d(s_gt,  gt_xy[:,0],  kind="linear")
    fN_gt   = interp1d(s_gt,  gt_xy[:,1],  kind="linear")

    P = np.stack([fE_pred(s_grid), fN_pred(s_grid)], axis=1)
    G = np.stack([fE_gt(s_grid),   fN_gt(s_grid)],   axis=1)

    rmse = np.sqrt(np.mean(np.sum((P - G)**2, axis=1)))
    return rmse

import numpy as np
from scipy.interpolate import interp1d


# ---------- helper: cumulative arc-length from x,y -----------------
def arc_length_xy(x, y):
    dx   = np.diff(x)
    dy   = np.diff(y)
    seg  = np.hypot(dx, dy)
    return np.concatenate(([0.0], np.cumsum(seg)))


# ---------- RMSE vs. distance, component-wise ----------------------
def rmse_vs_distance_xy_components(pred_E, pred_N,
                                   gt_E,   gt_N,
                                   ds=0.05):
    """
    Parameters
    ----------
    pred_E, pred_N : (N,) arrays – predicted Easting & Northing
    gt_E,   gt_N   : (M,) arrays – ground-truth Easting & Northing
    ds             : float – resample step (metres), default 0.05 m

    Returns
    -------
    rmse_E : float  (metres)  – RMSE in Easting
    rmse_N : float  (metres)  – RMSE in Northing
    """
    # cumulative distance along each path
    s_pred = arc_length_xy(pred_E, pred_N)
    s_gt   = arc_length_xy(gt_E,   gt_N)

    # common arc-length range
    s_max  = min(s_pred[-1], s_gt[-1])
    s_grid = np.arange(0.0, s_max, ds)
    # linear interpolation onto the common grid
    try:
        fE_pred = interp1d(s_pred, pred_E, kind="linear")
        fN_pred = interp1d(s_pred, pred_N, kind="linear")
        fE_gt   = interp1d(s_gt,   gt_E,   kind="linear")
        fN_gt   = interp1d(s_gt,   gt_N,   kind="linear")

        PE, PN = fE_pred(s_grid), fN_pred(s_grid)
        GE, GN = fE_gt(s_grid),   fN_gt(s_grid)

        rmse_E = np.sqrt(np.mean((PE - GE)**2))
        rmse_N = np.sqrt(np.mean((PN - GN)**2))
        return rmse_E, rmse_N
    except ValueError:
        return 0., 0.


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
                            vy_cpu[i] = [[scale_factor * dx, scale_factor * dy * 2] for dx, dy in vy_cpu[i]]
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

lx, ly = [], []
for idx, l in enumerate(labels):
    lx.append(l[0])
    ly.append(l[1])

lax, lay = [], []
for idx, l in enumerate(labels_actual):
    lax.append(l[0])
    lay.append(l[1])# with open(f"results/180sec.txt", "a") as f:
#     f.write("\n\n")
#     f.write(f"-------- e{str(dist_east / len(path))},n{str(dist_north / len(path))} --------\n")
#     for e, n in zip(east_list, north_list):
#         f.write(f"e{str(e)},n{str(n)}\n")
#     f.close()

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

print(preds_arr.shape, labs_arr.shape)

dist_east, _ = fastdtw(preds_arr[:,0], labs_arr[:,0], dist=euclidean)
dist_north, _ = fastdtw(preds_arr[:,1], labs_arr[:,1], dist=euclidean)

rmse = rmse_vs_distance(preds_arr, labs_arr)

print("Average path distance RMSE: ",rmse)

east_list, north_list = [], []
d_north, d_east = 0, 0
ii = 0
for i in range(labs_arr.shape[0]):
    dist_east, _ = fastdtw(preds_arr[-i:,0], labs_arr[-i:,0], dist=euclidean)
    dist_north, _ = fastdtw(preds_arr[-i:,1], labs_arr[-i:,1], dist=euclidean)
    
    east_list.append(dist_east / len(path))
    north_list.append(dist_north / len(path))
    
    d_east += dist_east
    d_north += dist_north

dist_east = d_east / labs_arr.shape[0]
dist_north = d_north / labs_arr.shape[1]

with open(f"results/180sec_acc.txt", "a") as f:
    f.write("\n\n")
    f.write(f"-------- e{str(dist_east / len(path))},n{str(dist_north / len(path))} --------\n")
    for e, n in zip(east_list, north_list):
        f.write(f"e{str(e)},n{str(n)}\n")
    f.close()

print(f"DTW distance between predicted East and actual East trajectories: {(dist_east / len(path)):.4f}")
print(f"DTW distance between predicted North and actual North trajectories: {(dist_north / len(path)):.4f}")

print(f"DTW distance between predicted and actual trajectories: {distance:.4f}")

# Optionally, you can compute average DTW distance per aligned pair:
avg_dtw_dist = distance / len(path)
print(f"Average DTW distance per aligned pair: {avg_dtw_dist:.4f}")

east_list, north_list = [], []
d_north, d_east = 0, 0
for i in range(labs_arr.shape[0]):
    dist_east, dist_north = rmse_vs_distance_xy_components(preds_arr[-i:,0], preds_arr[-i:,1],
                                   labs_arr[-i:,0], labs_arr[-i:,1],
                                   ds=0.05)
    
    east_list.append(dist_east / max(len(preds_arr[-i:,0]), len(labs_arr[-i:,0])))
    north_list.append(dist_north / max(len(preds_arr[-i:,1]), len(labs_arr[-i:,1])))
    
    d_east += dist_east
    d_north += dist_north

dist_east = d_east / labs_arr.shape[0]
dist_north = d_north / labs_arr.shape[1]

with open(f"results/180sec_acc_rmse.txt", "a") as f:
    f.write("\n\n")
    f.write(f"-------- e{str(dist_east / len(path))},n{str(dist_north / len(path))} --------\n")
    for e, n in zip(east_list, north_list):
        f.write(f"e{str(e)},n{str(n)}\n")
    f.close()

