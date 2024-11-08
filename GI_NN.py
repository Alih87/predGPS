import torch, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GI_NN(nn.Module):
    def __init__(self, input_size, output_channels, SEQ_LEN):
        super(GI_NN, self).__init__()
        self.input_size = input_size
        self.seq_len = SEQ_LEN
        self.gnn = nn.GRU(256, 128, 4, batch_first=True, bidirectional=True)
        
        self.first_layer = nn.Conv1d(self.input_size, 256, 1)
        self.pool1 = nn.AdaptiveAvgPool1d(SEQ_LEN // 2)

        self.second_layer = nn.Conv1d(256, 512, 1)
        self.pool2 = nn.AdaptiveAvgPool1d(SEQ_LEN // 4)

        self.third_layer = nn.Conv1d(512, 512, 1)
        self.pool3 = nn.AdaptiveAvgPool1d(SEQ_LEN // 8)

        self.fourth_layer = nn.Conv1d(512, 256, 1)
        self.pool4 = nn.AdaptiveAvgPool1d(SEQ_LEN // 16)

        self.fc = nn.Linear(256, 64)
        self.drop_out = nn.Dropout(0.15)
        self.last_layer = nn.Linear(64, output_channels)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.batch_norm_first = nn.BatchNorm1d(256)
        self.batch_norm_second = nn.BatchNorm1d(512)
        self.batch_norm_third = nn.BatchNorm1d(512)
        self.batch_norm_fourth = nn.BatchNorm1d(256)

    def forward(self, x):
        a = self.first_layer(x)
        a = self.batch_norm_first(a)
        a = self.relu(a)
        # a = self.pool1(a)

        a1 = self.second_layer(a)
        a1 = self.batch_norm_second(a1)
        a1 = self.relu(a1)
        # a1 = self.pool2(a1)

        a2 = self.third_layer(a1)
        a2 = self.batch_norm_third(a2)
        a2 = self.relu(a2)
        # a2 = self.pool3(a2)

        a3 = self.fourth_layer(a2)
        a3 = self.batch_norm_fourth(a3)
        a3 = self.relu(a3)
        # a3 = self.pool4(a3)
        a3 = a3.permute(0, 2, 1)
        
        b, _ = self.gnn(a3)
        print(b.shape)
        # c = self.fc(b[:, -1, :])
        c = self.fc(b)      # Use all of 'b' for many-to-many
        print(c.shape)
        c = self.tanh(c)
        d = self.drop_out(c)
        z = self.last_layer(d)
        
        z = torch.squeeze(z, dim=0)
        
        return z.cuda().float()

if __name__ == '__main__':
    os.chdir("/data_hdd1/hassan/projects/GPS")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN = 14
    INPUT_SIZE = 12

    model = GI_NN(input_size=INPUT_SIZE, output_channels=2, SEQ_LEN=SEQ_LEN)
    model.to(DEVICE)

    x = torch.randn(4, INPUT_SIZE, SEQ_LEN).to(DEVICE)
    out = model(x)
    # print(out.shape)
    