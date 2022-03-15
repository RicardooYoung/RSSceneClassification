import torch.nn as nn
import torch.nn.functional as f


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        y = self.block(x)
        y += self.shortcut(x)
        y = f.relu(y, inplace=True)
        return y


class ResNet34(nn.Module):
    def __init__(self, num_class):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            # size = /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # size = /4
        )
        self.block = nn.Sequential(
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            # size = /8
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1),
            ResBlock(128, 256, 2),
            # size = /16
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 512, 2),
            # size = /32
            ResBlock(512, 512, 1),
            ResBlock(512, 512, 1),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
