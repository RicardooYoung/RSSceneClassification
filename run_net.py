import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

import resnet
import trainer
import evaluator

batch_size = 96
# 96 for ResNet34
# 32 for ResNet50
train_path = 'Dataset/train'
test_path = 'Dataset/test'
train_set = ImageFolder(root=train_path, transform=ToTensor())
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
test_set = ImageFolder(root=test_path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=int(batch_size / 2), shuffle=False, num_workers=3, pin_memory=True)
# Load dataset.

# model = resnet.ResNet34(45)
# model = resnet.ResNet50(45)
model = resnet.PreResNet34(45)
# model = torch.load('model.pth')
if torch.cuda.is_available():
    model.cuda()
# Initialize CNN.

max_iteration = 30
lr = 1e-3
momentum = 0.9
weight_decay = 1e-4
# Define hyper-parameter

early_stop = 0

total_train_acc = np.zeros(max_iteration)
total_test_acc = np.zeros(max_iteration)
total_train_loss = np.zeros(max_iteration)
total_test_loss = np.zeros(max_iteration)
lr_curve = np.zeros(max_iteration)

if __name__ == '__main__':
    for epoch in range(max_iteration):
        total_train_acc[epoch], total_train_loss[epoch] = trainer.train_net(model, train_data, lr, momentum,
                                                                            weight_decay, epoch)
        torch.cuda.empty_cache()
        total_test_acc[epoch], total_test_loss[epoch] = evaluator.test_net(model, test_data, epoch)
        torch.cuda.empty_cache()

        if epoch > 19:
            temp = total_train_acc[epoch - 5:epoch + 1]
            max_acc = max(temp)
            min_acc = min(temp)
            if max_acc - min_acc < 1e-4:
                early_stop = epoch + 1
                break

        if epoch > 9:
            temp = total_train_acc[epoch - 2:epoch + 1]
            max_acc = max(temp)
            min_acc = min(temp)
            if max_acc - min_acc < 0.05:
                lr /= 10
                print('Learning rate changed.')

    torch.save(model, 'model.pth')

    if early_stop != 0:
        max_iteration = early_stop

    epoch = np.linspace(1, max_iteration, max_iteration)
    new_tick = np.linspace(0, max_iteration, max_iteration + 1)
    plt.figure()
    plt.plot(epoch, total_train_acc[0:max_iteration], '-.^')
    plt.plot(epoch, total_test_acc[0:max_iteration], '-.^')
    plt.title('Train & Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(new_tick)
    plt.legend(['Train Acc', 'Test Acc'])
    plt.savefig('fig1.jpg')
    plt.show()

    plt.figure()
    plt.plot(epoch, total_train_loss, '-.^')
    plt.plot(epoch, total_test_loss, '-.^')
    plt.title('Train & Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(new_tick)
    plt.legend(['Train Loss', 'Test Loss'])
    plt.savefig('fig2.jpg')
    plt.show()
