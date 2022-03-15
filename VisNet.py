import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import TestNet

path = 'Data'
dataset = ImageFolder(root=path, transform=ToTensor())
data = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
model = torch.load('model.pth')

if __name__ == '__main__':
    acc, loss = TestNet.test_net(model, data)
