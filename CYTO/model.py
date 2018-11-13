import torch
import torch.nn as nn

class ChannelBlock(nn.Module):
    def __init__(self):
        super(ChannelBlock, self).__init__()

        self.conv1 = nn.Sequential(
                     nn.Conv2d(1, 64, kernel_size=5, padding=1),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(2, padding=1)
        )

        self.conv2 = nn.Sequential(
                     nn.Conv2d(64, 128, kernel_size=3),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(2, padding=1)
        )
        self.conv3 = nn.Sequential(
                     nn.Conv2d(128, 256, kernel_size=3),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(2, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class CYTOModel(nn.Module):
    def __init__(self):
        super(CYTOModel, self).__init__()

        self.red = ChannelBlock()
        self.green = ChannelBlock()
        self.blue = ChannelBlock()
        self.yellow = ChannelBlock()

        self.conv1 = nn.Sequential(
                     nn.Conv2d(256, 256, kernel_size=3),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(2, padding=1)
        )

        self.conv2 = nn.Sequential(
                     nn.Conv2d(256, 28, kernel_size=3, padding=1),
                     nn.Sigmoid()
        )

    def forward(self, r, g, b, y):
        red = self.red(r)
        green = self.green(g)
        blue = self.red(b)
        yellow = self.red(y)

        features = red + green + blue + yellow
        features = self.conv1(features)
        features = self.conv2(features)
        return features
