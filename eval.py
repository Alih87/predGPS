import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
import wandb.plot
from GI_NN import GI_NN
from utils import IMUDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = "data/data.txt"
val_path = "data/val.txt"

SEQ_LEN = 24
INPUT_SIZE = 11

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

Xv, yv = extract_txt(train_path)

scaler_val = MinMaxScaler((-1,1))
yv = scaler_val.fit_transform(yv)

scaler_Xval = MinMaxScaler((-1,1))
Xv = scaler_Xval.fit_transform(Xv)

yv = (np.asanyarray(yv)).tolist()

validation_loader = IMUDataset(Xv, yv, seq_len=SEQ_LEN)


model = GI_NN(input_size=INPUT_SIZE, output_channels=3, SEQ_LEN=SEQ_LEN)
model.load_state_dict(torch.load("chkpts/20241104_145134/model_20241104_145134_53.pth"))
model.to(DEVICE)
model = model.cuda().float()
model.eval()

labels = []
preds = []
print("Epoch Done")
running_vloss = 0.0
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
            print(vy_.shape, vy.shape)
            vy_cpu, vycpu = vy_.cpu().tolist(), vy.cpu().tolist()
            for i in range(vy.shape[0]):
                preds.append([
                    vy_cpu[i][0] + preds[-1][0],
                    vy_cpu[i][1] + preds[-1][1],
                    vy_cpu[i][2] + preds[-1][2],
                    ])
                labels.append([
                    vycpu[i][0] + labels[-1][0],
                    vycpu[i][1] + labels[-1][1],
                    vycpu[i][2] + labels[-1][2],
                    ])
        except:
            print("[INFO] Not enough data, proceeding...")
            break

preds = scaler_val.inverse_transform(preds).tolist()
labels = scaler_val.inverse_transform(labels).tolist()

px, py = [], []
for idx, p in enumerate(preds):
    px.append(p[0][0])
    py.append(p[0][1])

lx, ly = [], []
for idx, l in enumerate(labels):
    lx.append(l[0][0])
    ly.append(l[0][1])

xx = 0
yy = 0
for p in preds:
    xx += p[0][0]
    yy += p[0][1]

plt.plot(px, py)
plt.show()

plt.plot(lx, ly)
plt.show()
