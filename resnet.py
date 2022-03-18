import torch.nn as nn
import torch.nn.functional as f


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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


class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
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
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            # size = /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # size = /4
        )
        self.block = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            # conv2_x
            ResBlock(64, 128, 2),
            # size = /8
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            # conv3_x
            ResBlock(128, 256, 2),
            # size = /16
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            # conv4_x
            ResBlock(256, 512, 2),
            # size = /32
            ResBlock(512, 512),
            ResBlock(512, 512),
            # conv5_x
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_class):
        super(ResNet50, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            # size = /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # size = /4
        )
        self.block = nn.Sequential(
            BottleNeck(64, 64, 256),
            BottleNeck(256, 64, 256),
            BottleNeck(256, 64, 256),
            # conv2_x
            BottleNeck(256, 128, 512, 2),
            # size = /8
            BottleNeck(512, 128, 512),
            BottleNeck(512, 128, 512),
            BottleNeck(512, 128, 512),
            BottleNeck(512, 128, 512),
            # conv3_x
            BottleNeck(512, 256, 1024, 2),
            # size = /16
            BottleNeck(1024, 256, 1024),
            BottleNeck(1024, 256, 1024),
            BottleNeck(1024, 256, 1024),
            BottleNeck(1024, 256, 1024),
            BottleNeck(1024, 256, 1024),
            # conv4_x
            BottleNeck(1024, 512, 2048, 2),
            # size = /32
            BottleNeck(2048, 512, 2048),
            BottleNeck(2048, 512, 2048),
            # conv5_x
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
