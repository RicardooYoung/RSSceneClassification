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


class PreResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreResUnit, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=1),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                          stride=stride)
            )

    def forward(self, x):
        y = self.block(x)
        y += self.shortcut(x)
        return y


class PreBottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(PreBottleNeck, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
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
        return y


class ResNet34(nn.Module):
    def __init__(self, num_class, metric_learn=False):
        super(ResNet34, self).__init__()
        self.metric_learn = metric_learn
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
        if not self.metric_learn:
            self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        if not self.metric_learn:
            x = self.fc(x)
        return x


class PreResNet34(nn.Module):
    def __init__(self, num_class, metric_learn=False):
        super(PreResNet34, self).__init__()
        self.metric_learn = metric_learn
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            # size = /2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # size = /4
        )
        self.block = nn.Sequential(
            PreResUnit(64, 64),
            PreResUnit(64, 64),
            PreResUnit(64, 64),
            # conv2_x
            PreResUnit(64, 128, 2),
            # size = /8
            PreResUnit(128, 128),
            PreResUnit(128, 128),
            PreResUnit(128, 128),
            # conv3_x
            PreResUnit(128, 256, 2),
            # size = /16
            PreResUnit(256, 256),
            PreResUnit(256, 256),
            PreResUnit(256, 256),
            PreResUnit(256, 256),
            PreResUnit(256, 256),
            # conv4_x
            PreResUnit(256, 512, 2),
            # size = /32
            PreResUnit(512, 512),
            PreResUnit(512, 512),
            # conv5_x
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        y = f.relu(x)
        y = self.fc(y)
        if self.metric_learn:
            return x, y
        else:
            return y

    def draw_feature(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = x.view(x.size()[0], -1)
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


class PreResNet50(nn.Module):
    def __init__(self, num_class):
        super(PreResNet50, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            # size = /2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # size = /4
        )
        self.block = nn.Sequential(
            PreBottleNeck(64, 64, 256),
            PreBottleNeck(256, 64, 256),
            PreBottleNeck(256, 64, 256),
            # conv2_x
            PreBottleNeck(256, 128, 512, 2),
            # size = /8
            PreBottleNeck(512, 128, 512),
            PreBottleNeck(512, 128, 512),
            PreBottleNeck(512, 128, 512),
            PreBottleNeck(512, 128, 512),
            # conv3_x
            PreBottleNeck(512, 256, 1024, 2),
            # size = /16
            PreBottleNeck(1024, 256, 1024),
            PreBottleNeck(1024, 256, 1024),
            PreBottleNeck(1024, 256, 1024),
            PreBottleNeck(1024, 256, 1024),
            PreBottleNeck(1024, 256, 1024),
            # conv4_x
            PreBottleNeck(1024, 512, 2048, 2),
            # size = /32
            PreBottleNeck(2048, 512, 2048),
            PreBottleNeck(2048, 512, 2048),
            # conv5_x
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = f.relu(x, inplace=True)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
