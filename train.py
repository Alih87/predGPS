import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
from GI_NN import GI_NN
from torch.utils.data import DataLoader
from dataloader import IMUDataset, collate_fn
import wandb, datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = "data/data.txt"
val_path = "data/val.txt"

SEQ_LEN = 14
INPUT_SIZE = 11
LAT_CENTER, LON_CENTER = 388731.70, 3974424.49

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

def train_one_epoch():
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
            y_ = model(X)
            loss = loss_fn(y_, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i%10 == 9:
                last_loss = running_loss / 1000
                print(f"Batch {i+1} loss: {last_loss}")
                running_loss = 0
        except:
            print("[INFO] Not enough data, proceeding...")
            break

    return last_loss

Xt, yt = extract_txt(train_path)
Xv, yv = extract_txt(val_path)

# scaler_ytrain = MinMaxScaler()
# scaler_yval = MinMaxScaler()

scaler_Xtrain = MinMaxScaler()
scaler_xval = MinMaxScaler()

# Xt = normalize_data(Xt)
# Xv = normalize_data(Xv)

scaler_Xtrain = MinMaxScaler()
scaler_Xval = MinMaxScaler()

Xt = scaler_Xtrain.fit_transform(Xt)
Xv = scaler_Xval.fit_transform(Xv)

# yt = scaler_ytrain.fit_transform(yt)
# yv = scaler_yval.fit_transform(yv)

yt = (np.asanyarray(yt)).tolist()
yv = (np.asanyarray(yv)).tolist()

training_loader = DataLoader(IMUDataset(Xt, yt, seq_len=SEQ_LEN, scaler=yt[0]), batch_size=128, shuffle=True)
validation_loader = DataLoader(IMUDataset(Xv, yv, seq_len=SEQ_LEN, scaler=yv[0]), batch_size=1, shuffle=False)

model = GI_NN(input_size=INPUT_SIZE, output_channels=2, SEQ_LEN=SEQ_LEN)
model.to(DEVICE)
model = model.cuda().float()
model.train()
optimizer = SGD(model.parameters(), lr = 1e-3, momentum=0.9)
loss_fn = nn.MSELoss()

if __name__ == '__main__':
    wandb.init(project="GNSS", entity='ciir')
    preds, labels = [], []
    EPOCH = 60
    train_loss, val_loss = [], []
    train_loss_all, val_loss_all = [], []
    epoch_number = 0
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_vloss = 99999
    for epoch in range(EPOCH):
        model.train()
        labels = []
        preds = []
        px, py = [], []
        lx, ly = [], []
        avg_loss = train_one_epoch()
        print("Epoch Done")
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                try:
                    vX, vy = vdata
                    vX = vX.cuda().float()
                    vy = vy.cuda().float()
                    vX.to(DEVICE)
                    vy.to(DEVICE)
                    vy_ = model(vX)
                    print(vy_, vy)
                    if len(vy_.shape) < 2:
                        vy_ = torch.unsqueeze(vy_, dim=0)
                    # labels.append(scaler_yval.inverse_transform(vy.cpu().tolist()))
                    # preds.append(scaler_yval.inverse_transform(vy_.cpu().tolist()))
                    labels.extend((vy.cpu()).tolist())
                    preds.extend((vy_.cpu()).tolist())
                    vloss = loss_fn(vy_, vy)
                    running_vloss += vloss

                except RuntimeError:
                    print("[INFO] Not enough data, proceeding...")
                    break
        
        # for idx, p in enumerate(preds):
        #     if idx == 0:
        #         px.append(p[0][0])
        #         py.append(p[0][1])
        #     else:
        #         px.append(px[idx-1] + p[0][0])
        #         py.append(py[idx-1] + p[0][1])
        #     px.append(p[0][0])
        #     py.append(p[0][1])
        
        # for idx, l in enumerate(labels):
        #     if idx == 0:
        #         lx.append(l[0][0])
        #         ly.append(l[0][1])
        #     else:
        #         lx.append(lx[idx-1] + l[0][0])
        #         ly.append(ly[idx-1] + l[0][1])
        #     lx.append(l[0][0])
        #     ly.append(l[0][1])

        # data1 = [[x,y] for (x,y) in zip(px, py)]
        # table1 = wandb.Table(data=data1, columns=["Easting", "Northing"])
        # wandb.log({"Predicted": wandb.plot.scatter(table1, "Easting", "Northing")})

        # data2 = [[x,y] for (x,y) in zip(lx, ly)]
        # table2 = wandb.Table(data=data2, columns=["Easting", "Northing"])
        # wandb.log({"Labels": wandb.plot.scatter(table2, "Easting", "Northing")})
            
        avg_vloss = running_vloss / (i+1)
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
        # if idx == 0:
        #     px.append(p[0][0])
        #     py.append(p[0][1])
        # else:
        #     px.append(px[idx-1] + p[0][0])
        #     py.append(py[idx-1] + p[0][1])
        px.append(p[0] + yv[0][0])
        py.append(p[1] + yv[0][1])
    
    lx, ly = [], []
    for idx, l in enumerate(labels):
        # if idx == 0:
        #     lx.append(l[0][0])
        #     ly.append(l[0][1])
        # else:
        #     lx.append(lx[idx-1] + l[0][0])
        #     ly.append(ly[idx-1] + l[0][1])
        lx.append(l[0] + yv[0][0])
        ly.append(l[1] + yv[0][1])

    data1 = [[x,y] for (x,y) in zip(px, py)]
    table1 = wandb.Table(data=data1, columns=["Easting", "Northing"])
    wandb.log({"Predicted": wandb.plot.scatter(table1, "Easting", "Northing")})

    data2 = [[x,y] for (x,y) in zip(lx, ly)]
    table2 = wandb.Table(data=data2, columns=["Easting", "Northing"])
    wandb.log({"Labels": wandb.plot.scatter(table2, "Easting", "Northing")})

    # xx = 0
    # yy = 0
    # for p in preds:
    #     xx += l[0][0]
    #     yy += l[0][1]

    # print(xx, yy)
    # plt.plot(train_loss)
    # plt.title("Training Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()

    # plt.plot(val_loss)
    # plt.title("Validation Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()

    os.mkdir(f"results/{time_stamp}")

    plt.title('Predicted')
    plt.plot(px, py)
    plt.savefig(f"results/{time_stamp}/predicted.png")

    plt.title('Labels')
    plt.plot(lx, ly)
    plt.savefig(f"results/{time_stamp}/labels.png")
