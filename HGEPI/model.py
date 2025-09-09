import math
from sklearn.metrics import roc_curve, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, roc_auc_score

metadata = (
    ['promoter', 'enhancer'], [('promoter', 'interactive', 'enhancer'), ('enhancer', 'rev_interactive', 'promoter')])


class CNN_promoter(torch.nn.Module):
    def __init__(self):
        super(CNN_promoter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=8, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 8, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, 4, 8, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
            nn.BatchNorm1d(4),
            nn.Dropout(0.35)  # 4 * 327
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 327, 1024)

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN_enhancer(torch.nn.Module):
    def __init__(self):
        super(CNN_enhancer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=8, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 8, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, 4, 8, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
            nn.BatchNorm1d(4),
            nn.Dropout(0.35)  # 4 * 493
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 493, 1024)

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class Bi_Linear_decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bilinear1 = nn.Bilinear(512, 512, 128)
        self.bilinear2 = nn.Bilinear(512, 512, 128)
        self.fc_r_0 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.fc_r_1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z, edge_index):
        x1 = z['promoter'][edge_index[0]]
        x2 = z['enhancer'][edge_index[1]]
        r_0 = self.bilinear1(x1, x2)
        r_0 = self.fc_r_0(r_0)
        r_1 = self.bilinear2(x1, x2)
        r_1 = self.fc_r_1(r_1)
        return torch.cat([r_0, r_1], dim=1)


class HAN(torch.nn.Module):
    def __init__(self, dp, dp2, hd1, hd2, hc, outc):
        super(HAN, self).__init__()
        self.han_conv1 = HANConv(1024, outc, heads=hd1,
                                 dropout=dp, metadata=metadata)
        self.lin1 = nn.Linear(outc, 512)
        self.relu = nn.ReLU()

    def forward(self, x_dict, edge_index_dict):
        x = self.han_conv1(x_dict, edge_index_dict)
        x['promoter'] = self.relu(x['promoter'])
        x['enhancer'] = self.relu(x['enhancer'])
        x['promoter'] = self.lin1(x['promoter'])
        x['enhancer'] = self.lin1(x['enhancer'])
        return x


class HGEPI(nn.Module):
    def __init__(self, dp, dp2, hd, hd2, hc, outc):
        super(HGEPI, self).__init__()
        self.cnn1 = CNN_promoter()
        self.cnn2 = CNN_enhancer()
        self.han = HAN(dp, dp2, hd, hd2, hc, outc)
        self.decoder = Bi_Linear_decoder()

    def forward(self, d_edge_list, data):
        x_dict = {}
        x_dict['promoter'] = self.cnn1(data.x_dict['promoter'])
        x_dict['enhancer'] = self.cnn2(data.x_dict['enhancer'])
        x = self.han(x_dict, data.edge_index_dict)
        # x = x_dict
        x = self.decoder(x, d_edge_list)
        return x

    def recon_loss(self, pred, y):
        loss = F.cross_entropy(pred, y)
        return loss

    def test_recon_loss(self, pred, y):
        y = y.long()
        pred = torch.nn.functional.softmax(pred, dim=1)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        pred = pred[:, -1]
        return y, pred
