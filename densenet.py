import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(DenseLayer, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        y = self.block(x)
        y = torch.cat((x, y), dim=1)
        return y


class DenseBlock(nn.Module):
    def __init__(self, in_channel, growth_rate, layer):
        super(DenseBlock, self).__init__()
        self.sequence = [in_channel + i * growth_rate for i in range(layer)]
        self.block = nn.Sequential(
            *[DenseLayer(self.sequence[i], growth_rate) for i in range(layer)]
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channel):
        super(Transition, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel / 2), kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, growth_rate, num_class):
        super(DenseNet121, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            # size = /2
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # size = /4
        )
        self.block = nn.Sequential(
            DenseBlock(2 * growth_rate, growth_rate, 6),
            Transition(8 * growth_rate),
            # size = /8
            DenseBlock(4 * growth_rate, growth_rate, 12),
            Transition(16 * growth_rate),
            # size = /16
            DenseBlock(8 * growth_rate, growth_rate, 24),
            Transition(32 * growth_rate),
            # size = /32
            DenseBlock(16 * growth_rate, growth_rate, 16),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(32 * growth_rate, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
