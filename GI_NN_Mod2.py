import torch, os
import torch.nn as nn
import torch.nn.functional as F
from utils import RecentAndFinalLoss, DirectionalGPSLoss, GPSLoss

class GI_NN(nn.Module):
    def __init__(self, input_size, output_channels, anchors, SEQ_LEN):
        super(GI_NN, self).__init__()
        self.input_size = input_size
        self.seq_len = SEQ_LEN
        self.anchors = anchors

        # Global feature extraction
        self.global_conv = nn.Conv1d(self.input_size, 256, 15, padding=7)
        self.mid_conv = nn.Conv1d(self.input_size, 128, 7, padding=3)
        self.fine_conv1 = nn.Conv1d(self.input_size, 64, 5, padding=2)
        self.fine_conv2 = nn.Conv1d(self.input_size, 64, 3, padding=1)

        # Batch normalization
        self.global_bn = nn.BatchNorm1d(256)
        self.mid_bn = nn.BatchNorm1d(128)
        self.fine_bn1 = nn.BatchNorm1d(64)
        self.fine_bn2 = nn.BatchNorm1d(64)

        # Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(SEQ_LEN // 2)
        self.mid_pool = nn.AdaptiveAvgPool1d(SEQ_LEN // 2)
        self.fine_pool = nn.MaxPool1d(2)

        # Fusion layers
        self.fusion_layer1 = nn.Conv1d(128 + 128, 128, kernel_size=1)
        self.fusion_bn1 = nn.BatchNorm1d(128)
        self.fusion_layer2 = nn.Conv1d(256 + 128, 256, kernel_size=1)
        self.fusion_bn2 = nn.BatchNorm1d(256)

        # GRU and output
        self.gnn = nn.GRU(256, 128, 4, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 64)
        self.drop_out = nn.Dropout(0.1)
        self.last_layer = nn.Linear(64, output_channels)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.loss_fn = RecentAndFinalLoss(anchors=self.anchors)

    def forward(self, x, y=None):
        # Global path
        g = self.relu(self.global_bn(self.global_conv(x)))
        g = self.global_pool(g)

        # Mid-level path
        m = self.relu(self.mid_bn(self.mid_conv(x)))
        m = self.mid_pool(m)

        # Fine-grained path
        f1 = self.relu(self.fine_bn1(self.fine_conv1(x)))
        f1 = self.fine_pool(f1)

        f2 = self.relu(self.fine_bn2(self.fine_conv2(x)))
        f2 = self.fine_pool(f2)

        # Fusion
        f_combined = torch.cat([f1, f2], dim=1)
        m_skip = m
        f_combined = self.relu(self.fusion_bn1(self.fusion_layer1(torch.cat([f_combined, m], dim=1)))) + m_skip

        g_skip = g
        combined = torch.cat([f_combined, g], dim=1)
        combined = self.relu(self.fusion_bn2(self.fusion_layer2(combined))) + g_skip

        # Prepare for GRU
        combined = combined.permute(0, 2, 1)

        # GRU
        b, _ = self.gnn(combined)
        c = self.fc(b) * 1.11
        # c = self.tanh(c)
        d = self.drop_out(c)
        z = self.last_layer(d)  # shape: [batch, seq_len, output_channels]

        if self.training:
            if y is None:
                raise ValueError("Targets cannot be None in training mode")
            z_flipped = torch.flip(z[:, -self.anchors:, :], [1])
            loss = self.loss_fn(z_flipped, y)
            return loss, z_flipped[:, -self.anchors:, :].cuda().float()
        else:
            z_flipped = torch.flip(z[:, -self.anchors:, :], [1])
            return z[:, -self.anchors:, :].cuda().float()


if __name__ == '__main__':
    os.chdir("/home/hassan/projects/predGPS")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN = 64
    INPUT_SIZE = 12
    ANCHORS = 32

    model = GI_NN(input_size=INPUT_SIZE, output_channels=2, anchors=ANCHORS, SEQ_LEN=SEQ_LEN)
    model.to(DEVICE)
    model.eval()
    x = torch.randn(4, INPUT_SIZE, SEQ_LEN).to(DEVICE)
    targets = torch.randn(4, ANCHORS, 2).to(DEVICE)
    out = model(x, targets)
    if model.training:
        out = model(x, targets)
        print("loss:", out[0])
        print("pred:", out[1])
    else:
        print(out)
    