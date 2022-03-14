import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

import ResNet
import TrainNet
import TestNet

train_path = 'Dataset/train'
test_path = 'Dataset/test'
train_set = ImageFolder(root=train_path, transform=ToTensor())
train_data = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_set = ImageFolder(root=test_path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
model = ResNet.ResNet18(256, 3, 45)
if torch.cuda.is_available():
    model.cuda()

max_iteration = 10
total_train_acc = np.zeros(max_iteration)
total_test_acc = np.zeros(max_iteration)

if __name__ == '__main__':
    for epoch in range(max_iteration):
        total_train_acc[epoch] = TrainNet.train_net(model, train_data, epoch, lr=1e-3, momentum=0.5)
        torch.cuda.empty_cache()
        total_test_acc[epoch] = TestNet.test_net(model, test_data, epoch)

    epoch = np.linspace(1, max_iteration, max_iteration)
    plt.figure()
    plt.plot(epoch, total_train_acc, marker='o')
    plt.plot(epoch, total_test_acc, marker='o')
    plt.legend(['Train Acc', 'Test Acc'])
    plt.show()
