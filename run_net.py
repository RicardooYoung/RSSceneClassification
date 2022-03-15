import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

import resnet
import trainer
import evaluator

train_path = 'Dataset/train'
test_path = 'Dataset/test'
train_set = ImageFolder(root=train_path, transform=ToTensor())
train_data = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=3)
test_set = ImageFolder(root=test_path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=3)
# Load dataset.

model = resnet.ResNet34(45)
if torch.cuda.is_available():
    model.cuda()
# Initialize CNN.

max_iteration = 20
lr = 1e-3
weight_decay = 1e-4
# Define hyper-parameter

total_train_acc = np.zeros(max_iteration)
total_test_acc = np.zeros(max_iteration)
total_train_loss = np.zeros(max_iteration)
total_test_loss = np.zeros(max_iteration)

if __name__ == '__main__':
    for epoch in range(max_iteration):
        total_train_acc[epoch], total_train_loss[epoch] = trainer.train_net(model, train_data, epoch, lr=lr,
                                                                            momentum=0.9, weight_decay=weight_decay)
        torch.cuda.empty_cache()
        total_test_acc[epoch], total_test_loss[epoch] = evaluator.test_net(model, test_data, epoch)
        torch.cuda.empty_cache()

        if epoch >= 5 and abs(total_train_loss[epoch] - total_train_loss[epoch - 1]) < lr * 1e+2:
            if lr > 1e-5:
                lr /= 10
                print('Learning rate changed.')
            else:
                print('Learning rate is too small to be changed.')
        # If error rate plateaus, decrease the learning rate.
        if epoch >= 5 and total_test_loss[epoch] > total_test_loss[epoch - 1] > total_test_loss[epoch - 2]:
            if weight_decay <= 5 * 1e-4:
                weight_decay += 1e-4
                print('Weight decay changed.')
            else:
                print('Weight decay is too big to be changed.')
        if epoch >= 5 and total_train_acc[epoch] - total_test_acc[epoch] >= 0.1 * (1e-4 / weight_decay):
            if weight_decay <= 5 * 1e-4:
                weight_decay += 1e-4
                print('Weight decay changed.')
            else:
                print('Weight decay is too big to be changed.')
        # Roughly detect if over-fitting occurred.

    torch.save(model, 'model.pth')

    epoch = np.linspace(1, max_iteration, max_iteration)
    new_tick = np.linspace(0, 20, 11)
    plt.figure()
    plt.plot(epoch, total_train_acc, '-.^')
    plt.plot(epoch, total_test_acc, '-.^')
    plt.title('Train & Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(new_tick)
    plt.legend(['Train Acc', 'Test Acc'])
    plt.show()
