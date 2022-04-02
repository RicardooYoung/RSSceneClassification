import torch.nn as nn
import torch.nn.functional as f


class DenseBlock(nn.Module):
    def __init__(self, growth_rate):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Conv2d() for i in range(5)]
        )


class DenseNet101(nn.Module):
    def __init__(self, growth_rate):
        super(DenseNet101, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=growth_rate, kernel_size=7, stride=2, padding=3),
            # size = /2
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # size = /4
        )
