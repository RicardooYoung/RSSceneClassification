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
train_data = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=3)
test_set = ImageFolder(root=test_path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
model = ResNet.ResNet34(45)
if torch.cuda.is_available():
    model.cuda()

max_iteration = 20
lr = 1e-3
total_train_acc = np.zeros(max_iteration)
total_test_acc = np.zeros(max_iteration)
total_train_loss = np.zeros(max_iteration)
total_test_loss = np.zeros(max_iteration)

if __name__ == '__main__':
    for epoch in range(max_iteration):
        total_train_acc[epoch], total_train_loss[epoch] = TrainNet.train_net(model, train_data, epoch, lr=lr,
                                                                             momentum=0.9, weight_decay=1e-4)
        torch.cuda.empty_cache()
        total_test_acc[epoch], total_test_loss[epoch] = TestNet.test_net(model, test_data, epoch)
        torch.cuda.empty_cache()
        if epoch >= 1 and abs(total_train_loss[epoch] - total_train_loss[epoch - 1]) < 1e-2:
            lr /= 10

    # torch.save(model, 'model.pth')

    epoch = np.linspace(1, max_iteration, max_iteration)
    plt.figure()
    plt.plot(epoch, total_train_acc, '-.^')
    plt.plot(epoch, total_test_acc, '-.^')
    plt.title('Train & Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Acc', 'Test Acc'])
    plt.show()
