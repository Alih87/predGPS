import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from GI_NN_Mod import GI_NN
from torch.utils.data import DataLoader
from utils import IMUDataset, DirectionalGPSLoss, GPSLoss, RecentAndFinalLoss

import wandb, datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_path = "data/data.txt"
val_path = "data/val.txt"

SEQ_LEN = 30
INPUT_SIZE = 14
LAT_CENTER, LON_CENTER = 388731.70, 3974424.49
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_VAL = 512
ANCHORS = 14     # usually 10

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

def train_one_epoch(anchors):
    running_loss = 0.0
    last_loss = 0.0
    for i, data in enumerate(training_loader):
        try:
            X, y = data
            X = X.cuda().float()
            y = y.cuda().float()
            X.to(DEVICE)
            y.to(DEVICE)
            optimizer.zero_grad()
            loss, y_ = model(X, y)
            loss.backward()

            optimizer.step()
            print(f"Batch number {i+1} loss: {loss}")
            running_loss += loss.item()
        except:
            print("[INFO] Not enough data, proceeding...")
            break
    last_loss = running_loss / (i+1)
    return last_loss

Xt, yt = extract_txt(train_path)
Xv, yv = extract_txt(val_path)

scaler_Xtrain = MinMaxScaler((-1,1))
scaler_Xval = MinMaxScaler((-1,1))
Xt = scaler_Xtrain.fit_transform(Xt)
Xv = scaler_Xval.fit_transform(Xv)

scaler_ytrain = MinMaxScaler((-1,1))
scaler_yval = MinMaxScaler((-1,1))
yt = scaler_ytrain.fit_transform(yt)
yv = scaler_yval.fit_transform(yv)
yt = (np.asanyarray(yt)).tolist()
yv = (np.asanyarray(yv)).tolist()

training_loader = DataLoader(IMUDataset(Xt, yt, seq_len=SEQ_LEN, anchors=ANCHORS, scaler=yt[0]), batch_size=BATCH_SIZE_TRAIN, shuffle=True)
validation_loader = DataLoader(IMUDataset(Xv, yv, seq_len=SEQ_LEN, anchors=None, scaler=yv[0]), batch_size=BATCH_SIZE_VAL, shuffle=False)

model = GI_NN(input_size=INPUT_SIZE, output_channels=2, anchors=ANCHORS, SEQ_LEN=SEQ_LEN)
model.to(DEVICE)
model = model.cuda().float()
model.train()
optimizer = Adam(model.parameters(), lr = 1e-3)
loss_fn = RecentAndFinalLoss(anchors=ANCHORS)

if __name__ == '__main__':
    wandb.init(project="GNSS", entity='ciir')
    preds, labels = [], []
    EPOCH = 400
    train_loss, val_loss = [], []
    train_loss_all, val_loss_all = [], []
    epoch_number = 0
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_vloss = 99999
    for epoch in range(EPOCH):
        model.train()
        labels, preds = list(), list()
        preds.append([0,0])
        labels.append([0,0])
        px, py = [], []
        lx, ly = [], []
        avg_loss = train_one_epoch(anchors=ANCHORS)
        print("Epoch Done")
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for ii, vdata in enumerate(validation_loader):
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
                    vy_cpu, vycpu = vy_.cpu().tolist(), vy.cpu().tolist()
                    for i in range(vy.shape[0]):
                        preds.append([
                            vy_cpu[i][0] + preds[-1][0],
                            vy_cpu[i][1] + preds[-1][1]
                                     ])
                        labels.append([
                            vycpu[i][0] + labels[-1][0],
                            vycpu[i][1] + labels[-1][1]
                                      ])
                    vloss = loss_fn(vy_, vy)
                    print(f"Batch number {ii+1} loss: {vloss}")
                    running_vloss += vloss.item()

                except RuntimeError:
                    print("[INFO] Not enough data, proceeding...")
        preds = scaler_ytrain.inverse_transform(preds).tolist()
        labels = scaler_ytrain.inverse_transform(labels).tolist()
        avg_vloss = running_vloss / (ii+1)
        print(f"[EPOCH {epoch}] Loss train {avg_loss}, validation {avg_vloss}")
        train_loss.append(avg_loss)
        val_loss.append(avg_vloss)

        wandb.log({
            "Training": avg_loss,
            "Validation": avg_vloss
        })
    
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            os.makedirs(f"chkpts/{time_stamp}", exist_ok=True)
            model_path = f"chkpts/{time_stamp}/model_{time_stamp}_{epoch_number}.pth"
            torch.save(model.state_dict(), model_path)
        
        epoch_number += 1
    
    px, py = [], []
    for idx, p in enumerate(preds):
        px.append(p[0])
        py.append(p[1])
    
    lx, ly = [], []
    for idx, l in enumerate(labels):
        lx.append(l[0])
        ly.append(l[1])

    data1 = [[x,y] for (x,y) in zip(px, py)]
    table1 = wandb.Table(data=data1, columns=["Easting", "Northing"])
    wandb.log({"Predicted": wandb.plot.scatter(table1, "Easting", "Northing")})

    data2 = [[x,y] for (x,y) in zip(lx, ly)]
    table2 = wandb.Table(data=data2, columns=["Easting", "Northing"])
    wandb.log({"Labels": wandb.plot.scatter(table2, "Easting", "Northing")})

    os.mkdir(f"results/{time_stamp}")

    with open(f"results/{time_stamp}/preds.txt", 'w') as f:
        for x,y in zip(px,py):
            f.write(str(x)+","+str(y)+"\n")
        f.close()

    plt.title('Trajectory Comparison')
    plt.plot(px, py)
    plt.plot(lx, ly)
    plt.legend(["Predicted", "Labels"])
    plt.savefig(f"results/{time_stamp}/trajectories.png")
