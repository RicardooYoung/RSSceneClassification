import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import os

import resnet
import trainer
import evaluator

train_path = 'Dataset/train'
validation_path = 'Dataset/validation'
train_set = ImageFolder(root=train_path, transform=ToTensor())
validation_set = ImageFolder(root=validation_path, transform=ToTensor())

if not os.path.exists('Result'):
    os.mkdir('Result')

# model = torch.load('resnet34.pth')
# model.metric_learn = True
# model.cpu()
#
# proxy = np.zeros(45)
# n = 140
# idx = np.arange(n)
# if not os.path.exists('proxy.pth'):
#     if not os.path.exists('proxy.npy'):
#         with torch.no_grad():
#             for i in range(45):
#                 temp_set = torch.utils.data.Subset(train_set, idx)
#                 temp_data = DataLoader(temp_set, batch_size=n, shuffle=False)
#                 for image, label in temp_data:
#                     out, _ = model(image)
#                 dist = torch.pow(out, 2).sum(dim=1, keepdim=True).expand(n, n)
#                 dist = dist + dist.t()
#                 dist = torch.addmm(1, dist, -2, out, out.t())
#                 dist = dist.clamp(min=1e-12).sqrt()
#                 dist = dist.sum(axis=0)
#                 flag = torch.argmin(dist)
#                 flag = flag.item()
#                 proxy[i] = i * n + flag
#                 print('Proxy for class {} has been chosen.'.format(i))
#                 idx += n
#                 np.save('proxy.npy', proxy)
#     else:
#         proxy = np.load('proxy.npy')
#
#     proxy = proxy.astype(int)
#     temp_set = torch.utils.data.Subset(train_set, proxy)
#     temp_data = DataLoader(temp_set, batch_size=45, shuffle=False)
#     for image, _ in temp_data:
#         proxy, _ = model(image)
#     torch.save(proxy, 'proxy.pth')
# else:
#     proxy = torch.load('proxy.pth')
#
# proxy.detach_()
#
# if torch.cuda.is_available():
#     model.cuda()
#     proxy = proxy.cuda()

model = resnet.PreResNet34(45, True)

alpha = 0.05
lr = 1e-5
momentum = 0.9
max_iteration = 10
weight_decay = 0.005
batch_size = 64
# Define hyper-parameter


train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
validation_data = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

total_train_acc = np.zeros(max_iteration)
total_validation_acc = np.zeros(max_iteration)
total_train_loss = np.zeros(max_iteration)
total_validation_loss = np.zeros(max_iteration)

if __name__ == '__main__':
    for epoch in range(max_iteration):
        total_train_acc[epoch], total_train_loss[epoch] = trainer.train_net(model, train_data, optimizer, epoch,
                                                                            alpha=0, metric_learn=False, proxy=proxy)
        total_validation_acc[epoch], total_validation_loss[epoch] = evaluator.test_net(model, validation_data,
                                                                                       epoch, metric_learn=True)
    torch.save(model, 'resnet34_proxy.pth')
