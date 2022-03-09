import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

import ResNet


def run_net(model, train_data, test_data, lr=1e-1, momentum=0.0, weight_decay=0.0, it=10):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    total_train_acc = np.zeros(it)
    total_test_acc = np.zeros(it)

    for epoch in range(it):

        print('Epoch {} start.'.format(epoch + 1))

        time_start = time.time()
        train_loss = 0
        train_acc = 0
        model.train()

        for image, label in train_data:
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            out = model(image)
            loss = loss_fn(out, label)

            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / image.shape[0]
            train_acc += acc

        total_train_acc[epoch] = train_acc / len(train_data)
        time_end = time.time()

        print(
            'Epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Time elapsed: {:.3f}s'
                .format(epoch + 1, train_loss / len(train_data), train_acc / len(train_data), time_end - time_start))

        time_start = time.time()
        test_loss = 0
        test_acc = 0

        for image, label in test_data:
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            out = model(image)
            loss = loss_fn(out, label)

            test_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / image.shape[0]
            test_acc += acc

        total_test_acc[epoch] = test_acc / len(test_data)
        time_end = time.time()

        print(
            'Epoch: {}, Test Loss: {:.6f}, Test Acc: {:.6f}, Time elapsed: {:.3f}s'
                .format(epoch + 1, test_loss / len(test_data), test_acc / len(test_data), time_end - time_start))

    epoch = np.linspace(1, it, it)
    plt.figure()
    plt.plot(epoch, total_train_acc, marker='o')
    plt.plot(epoch, total_test_acc, marker='o')
    plt.legend(['Train Acc', 'Test Acc'])
    plt.show()


train_path = 'Dataset/train'
test_path = 'Dataset/test'
train_set = ImageFolder(root=train_path, transform=ToTensor())
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = ImageFolder(root=test_path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=64, shuffle=False)
print('Dataset created.')
model = ResNet.ResNet18(256, 3, 45)
if torch.cuda.is_available():
    model.cuda()
    print('Run on CUDA.')
print('Model loaded.')

run_net(model, train_data, test_data, lr=1e-3, momentum=0.1, weight_decay=0.1, it=10)
