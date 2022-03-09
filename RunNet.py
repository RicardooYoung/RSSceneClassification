import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import ResNet


def run_net(model, train_data, test_data, lr=1e-1, momentum=0.75, it=10):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    total_test_acc = np.zeros(it)

    # dic = np.zeros((45, 6))
    # for i in range(45):
    #     dic[i] = dec2bin(i)

    for epoch in range(it):

        print('Epoch {} start.'.format(epoch + 1))

        time_start = time.time()
        model.train()
        print('Start training.')

        for image, label in train_data:
            image = image.cuda()
            label = label.cuda()
            out = model(image)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = 0
        test_acc = 0
        model.eval()
        print('Start evaluating.')

        for image, label in test_data:
            image = image.cuda()
            label = label.cuda()
            out = model(image)
            loss = loss_fn(out, label)

            test_loss += loss.item()

            # pred = distance(out, dic)
            # label = bin2dec(label)
            pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / image.shape[0]
            test_acc += acc

        total_test_acc[epoch] = test_acc / len(test_data)
        time_end = time.time()

        print(
            'epoch: {}, Test Loss: {:.6f}, Test Acc: {:.6f}, Time elapsed: {:.3f}s'
                .format(epoch + 1, test_loss / len(test_data), test_acc / len(test_data), time_end - time_start))

    # epoch = np.linspace(1, it, it)
    # plt.figure()
    # plt.plot(epoch, total_test_acc, marker='o')
    # plt.legend(['Test Acc'])
    # plt.show()


path = 'Data'
dataset = ImageFolder(root=path, transform=ToTensor())
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=64, shuffle=False)
print('Dataset loaded.')
model = ResNet.ResNet18(256, 3, 45)
model.cuda()
print('Model loaded.')
run_net(model, train_data, test_data, lr=1e-1, momentum=0.75, it=10)
