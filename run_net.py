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
train_path = 'Dataset/train'
test_path = 'Dataset/test'
train_set = ImageFolder(root=train_path, transform=ToTensor())
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
test_set = ImageFolder(root=test_path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=int(batch_size / 2), shuffle=False, num_workers=3)
# Load dataset.

model = resnet.ResNet34(45)
# model = resnet.ResNet50(45)
if torch.cuda.is_available():
    model.cuda()
# Initialize CNN.

max_iteration = 20
lr = 1e-3
weight_decay = 1e-4
flag = 0
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

        if epoch != 0 and epoch % 4 == 0:
            lr /= 2
            print('Learning rate changed.')

        if total_train_acc[epoch] - total_test_acc[epoch] >= 0.15:
            if weight_decay <= 5 * 1e-4 and flag == 0:
                weight_decay += 1e-4
                flag = 1
                print('Weight decay changed.')
        else:
            flag = 0
            # Prevent change weight decay continuously

    torch.save(model, 'model.pth')

    epoch = np.linspace(1, max_iteration, max_iteration)
    new_tick = np.linspace(0, max_iteration, max_iteration + 1)
    plt.figure()
    plt.plot(epoch, total_train_acc, '-.^')
    plt.plot(epoch, total_test_acc, '-.^')
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
