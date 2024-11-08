import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
import wandb.plot
from GI_NN_Mod import GI_NN
from utils import IMUDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

train_path = "data/data.txt"
val_path = "data/val.txt"

SEQ_LEN = 38
INPUT_SIZE = 14
ANCHOR = 7

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

yv = (np.asanyarray(yv)).tolist()
yt = (np.asanyarray(yt)).tolist()

validation_loader = DataLoader(IMUDataset(Xv, yv, seq_len=SEQ_LEN, anchors=ANCHOR), batch_size=256, shuffle=False)


model = GI_NN(input_size=INPUT_SIZE, output_channels=2, anchors=ANCHOR, SEQ_LEN=SEQ_LEN)
model.load_state_dict(torch.load("chkpts/38SEQ_WeightedHuber_ANCHOR/model_20241108_063857_68.pth"))
# model.load_state_dict(torch.load("chkpts/20241105_165051/model_20241105_165051_61.pth"))
model.to(DEVICE)
model = model.cuda().float()
model.eval()

print("Epoch Done")
running_vloss = 0.0
labels, preds = list(), list()
preds.append([0,0])
labels.append([0,0])
with torch.no_grad():
    for i, vdata in enumerate(validation_loader):
        try:
            vX, vy = vdata
            vX = vX.cuda().float()
            vy = vy.cuda().float()
            vX.to(DEVICE)
            vy.to(DEVICE)
            vy_ = model(vX)
            if len(vy_.shape) < 2 or len(vy.shape) < 2:
                vy_ = torch.unsqueeze(vy_, dim=0)
                vy = torch.unsqueeze(vy, dim=0)
            if ANCHOR is not None:
                vy = vy[:,-1,:].squeeze(dim=1)
            vy_cpu, vycpu = vy_.cpu().tolist(), vy.cpu().tolist()
            for i in range(vy.shape[0]):
                preds.append([
                    vy_cpu[i][0] + preds[-1][0],
                    # preds[-1][0] - vy_cpu[i][0],
                    vy_cpu[i][1] + preds[-1][1]
                    # preds[-1][1] - vy_cpu[i][1]
                    ])
                labels.append([
                    vycpu[i][0] + labels[-1][0],
                    vycpu[i][1] + labels[-1][1],
                    ])
        except:
            print("[INFO] Not enough data, proceeding...")
            break

preds = scaler_val.inverse_transform(preds).tolist()
labels = scaler_val.inverse_transform(labels).tolist()

px, py = [], []
for idx, p in enumerate(preds):
    px.append(p[0])
    py.append(p[1])

lx, ly = [], []
for idx, l in enumerate(labels):
    lx.append(l[0])
    ly.append(l[1])

plt.plot(px, py)
plt.plot(lx, ly)
plt.title("Predicted vs Actual")
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.legend(["Predicted", "Actual"])
plt.show()
