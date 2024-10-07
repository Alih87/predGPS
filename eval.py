import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
import wandb.plot
from GI_NN import GI_NN
from dataloader import IMUDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = "data/data.txt"
val_path = "data/val.txt"

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
Xv = normalize_data(Xv)

scaler_val = MinMaxScaler()
yv = scaler_val.fit_transform(yv)

validation_loader = IMUDataset(Xv, yv, seq_len=7)


model = GI_NN(output_channels=2)
model.load_state_dict(torch.load("chkpts/20240927_144411/model_20240927_144411_19.pth"))
model.to(DEVICE)
model = model.cuda().float()
model.eval()

labels = []
preds = []
print("Epoch Done")
running_vloss = 0.
with torch.no_grad():
    for i, vdata in enumerate(validation_loader):
        try:
            vX, vy = vdata
            vX = vX.cuda().float()
            vy = vy.cuda().float()
            vX.to(DEVICE)
            vy.to(DEVICE)
            vy_ = model(vX)
            labels.append(scaler_val.inverse_transform([vy.cpu().tolist()]))
            preds.append(scaler_val.inverse_transform([vy_.cpu().tolist()]))
        except:
            print("[INFO] Not enough data, proceeding...")
            break
    
px, py = [], []
for idx, p in enumerate(preds):
    if idx == 0:
        print(p)
        px.append(p[0][0])
        py.append(p[0][1])
    else:
        px.append(px[idx-1] + p[0][0])
        py.append(py[idx-1] + p[0][1])

lx, ly = [], []
for idx, l in enumerate(labels):
    if idx == 0:
        lx.append(l[0][0])
        ly.append(l[0][1])
    else:
        lx.append(lx[idx-1] + l[0][0])
        ly.append(ly[idx-1] + l[0][1])

xx = 0
yy = 0
for p in preds:
    xx += p[0][0]
    yy += p[0][1]

print(xx, yy)

plt.plot(px, py)
plt.show()

plt.plot(lx, ly)
plt.show()
