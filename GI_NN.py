import torch, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torchviz import make_dot

class GI_NN(nn.Module):
    def __init__(self, input_size, output_channels, SEQ_LEN):
        super(GI_NN, self).__init__()
        self.input_size = input_size
        self.seq_len = SEQ_LEN
        self.gnn = nn.GRU(256, 128, 30, batch_first=True, bidirectional=False)
        self.first_layer = nn.Conv1d(self.input_size, 256, 1)
        self.fc = nn.Linear(128, 64)
        self.drop_out = nn.Dropout(0.1)
        self.last_layer = nn.Linear(64, output_channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[-1]):
        #     x_list.append(self.first_layer(torch.unsqueeze(x[:,:,i], dim=2)))
        # a = torch.cat(tuple(x_list), 2)
        # a = torch.unsqueeze(torch.sum(a, 1), 0)
        # print(a.shape)
        # x = x.transpose(1, 2)
        a = self.first_layer(x)
        a = self.relu(a)
        a = a.permute(0, 2, 1)
        # print(a.shape)
        b = self.gnn(a)
        # print(b[0].shape)
        c = self.fc(torch.mean(b[0],dim=1))
        c = self.relu(c)
        # print(c.shape)
        d = self.drop_out(c)
        # print(d.shape)
        z = self.last_layer(d)
        # print(z.shape)
        z = torch.squeeze(z, dim=0)
        return z.cuda().float()

if __name__ == '__main__':
    os.chdir("/data_hdd1/hassan/projects/GPS")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN = 7
    INPUT_SIZE = 9

    model = GI_NN(input_size=INPUT_SIZE, output_channels=2, SEQ_LEN=SEQ_LEN)
    model.to(DEVICE)

    x = torch.randn(4, 9, SEQ_LEN).to(DEVICE)
    out = model(x)
    print(out)
    # dot = make_dot(out.mean(), params=dict(model.named_parameters()))
    # dot.format ="png"
    # dot.render("model_arch")
    