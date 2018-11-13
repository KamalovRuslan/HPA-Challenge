import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

from tqdm import tqdm_notebook
from sklearn.metrics import f1_score


import torch.optim as optim
from torch.autograd import Variable
import torch

class GapNet(nn.Module):
    def __init__(self):
        super(GapNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2)
        self.mp1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.mp2 = nn.MaxPool2d(3, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1)

        self.classifier = nn.Sequential(
                                        nn.Linear(224, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(256, 28),
                                        nn.Sigmoid()
                                       )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        gap1 = x.max(dim=2)[0].max(dim=2)[0]

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp2(x)
        gap2 = x.max(dim=2)[0].max(dim=2)[0]

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        gap3 = x.max(dim=2)[0].max(dim=2)[0]

        features = torch.cat([gap1, gap2, gap3], dim=1)
        probs = self.classifier(features)
        return probs

def f1_loss(y_true, y_pred):
    tp = y_true * y_pred
    tp = tp.float().sum(dim=0)

    tn = (1. - y_true) * (1. - y_pred)
    tn = tn.float().sum(dim=0)

    fp = (1. - y_true) * y_pred
    fp = fp.float().sum(dim = 0)

    fn = y_true * (1. - y_pred)
    fn = fn.float().sum(dim=0)


    p = tp / (tp + fp + 1e-5)
    r = tp / (tp + fn + 1e-5)

    f1 = 2*p*r / (p+r+1e-5)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)
