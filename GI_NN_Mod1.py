import torch, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import RecentAndFinalLoss

class GI_NN(nn.Module):
    def __init__(self, input_size, output_channels, anchors, SEQ_LEN):
        super(GI_NN, self).__init__()
        self.input_size = input_size
        self.seq_len = SEQ_LEN
        self.anchors = anchors

        # Global feature extraction (larger kernel size)
        self.global_conv = nn.Conv1d(self.input_size, 256, 15, padding=7)
        
        # Mid-level feature extraction
        self.mid_conv = nn.Conv1d(self.input_size, 128, 7, padding=3)

        # Fine-grained detail extraction (small kernels)
        self.fine_conv1 = nn.Conv1d(self.input_size, 64, 5, padding=2)
        self.fine_conv2 = nn.Conv1d(self.input_size, 64, 3, padding=1)

        # Batch normalization for each module
        self.global_bn = nn.BatchNorm1d(256)
        self.mid_bn = nn.BatchNorm1d(128)
        self.fine_bn1 = nn.BatchNorm1d(64)
        self.fine_bn2 = nn.BatchNorm1d(64)

        # Pooling layers (flexible approach with max pooling for small kernels)
        self.global_pool = nn.AdaptiveAvgPool1d(SEQ_LEN // 2)
        self.mid_pool = nn.AdaptiveAvgPool1d(SEQ_LEN // 2)
        self.fine_pool = nn.MaxPool1d(2)

        # Progressive fusion layers
        self.fusion_layer1 = nn.Conv1d(128 + 128, 128, kernel_size=1)  # Fuse fine and mid features
        self.fusion_bn1 = nn.BatchNorm1d(128)

        self.fusion_layer2 = nn.Conv1d(256 + 128, 256, kernel_size=1)  # Fuse fused features with global
        self.fusion_bn2 = nn.BatchNorm1d(256)

        # GRU and fully connected layers
        self.gnn = nn.GRU(256, 128, 4, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 64)
        self.drop_out = nn.Dropout(0.35)
        self.last_layer = nn.Linear(64, output_channels)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.loss_fn = RecentAndFinalLoss(self.anchors)

    def forward(self, x, y=None):
        # Global path (captures broad trends)
        g = self.global_conv(x)
        g = self.global_bn(g)
        g = self.relu(g)
        g = self.global_pool(g)

        # Mid-level features (captures intermediate patterns)
        m = self.mid_conv(x)
        m = self.mid_bn(m)
        m = self.relu(m)
        m = self.mid_pool(m)

        # Fine-grained features (captures intricate details)
        f1 = self.fine_conv1(x)
        f1 = self.fine_bn1(f1)
        f1 = self.relu(f1)
        f1 = self.fine_pool(f1)

        f2 = self.fine_conv2(x)
        f2 = self.fine_bn2(f2)
        f2 = self.relu(f2)
        f2 = self.fine_pool(f2)

        # Concatenate fine-grained features and fuse with mid-level features
        f_combined = torch.cat([f1, f2], dim=1)
        m_skip = m  # Skip connection for mid-level features
        f_combined = self.fusion_layer1(torch.cat([f_combined, m], dim=1))
        f_combined = self.fusion_bn1(f_combined)
        f_combined = self.relu(f_combined) + m_skip  # Add skip connection

        # Fuse the combined fine+mid features with global features
        g_skip = g  # Skip connection for global features
        combined = torch.cat([f_combined, g], dim=1)
        combined = self.fusion_layer2(combined)
        combined = self.fusion_bn2(combined)
        combined = self.relu(combined) + g_skip  # Add skip connection

        # Prepare for GRU by permuting
        combined = combined.permute(0, 2, 1)

        # GRU and output layers
        b, _ = self.gnn(combined)
        if self.anchors is not None:
            c = self.fc(b[:, -1*self.anchors:, :])  # Many-to-many output
        else:
            c = self.fc(b[:, -1, :])  # Many-to-one output
        c = self.tanh(c)
        d = self.drop_out(c)
        z = self.last_layer(d)

        if self.training:
            if y is None:
                raise ValueError("Targets cannot be None in training mode")
            loss = self.loss_fn(z, y)
            return loss, z.cuda().float()
        else:
            return z[:, -1*self.anchors:, :] if self.anchors is not None else torch.squeeze(z, dim=0).cuda().float()

if __name__ == '__main__':
    os.chdir("/data_hdd1/hassan/projects/GPS")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN = 48
    INPUT_SIZE = 12
    ANCHORS = 9

    model = GI_NN(input_size=INPUT_SIZE, output_channels=2, anchors=ANCHORS, SEQ_LEN=SEQ_LEN)
    model.to(DEVICE)
    model.train()
    x = torch.randn(4, INPUT_SIZE, SEQ_LEN).to(DEVICE)
    targets = torch.randn(4, ANCHORS, 2).to(DEVICE)
    out = model(x, targets)
    if model.training:
        out = model(x, targets)
        print("loss:", out[0])
        print("pred:", out[1])
    