import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import LambdaLR
import os

import densenet
import resnet
import trainer
import evaluator

train_path = 'Dataset/train'
validation_path = 'Dataset/validation'
train_set = ImageFolder(root=train_path, transform=ToTensor())
validation_set = ImageFolder(root=validation_path, transform=ToTensor())

if not os.path.exists('Result'):
    os.mkdir('Result')

model_sequence = ['resnet34_m', 'densenet121_m']
for chosen_model in model_sequence:
    if chosen_model == 'resnet34_m':
        model = resnet.PreResNet34(45, True)
        batch_size = 96
    elif chosen_model == 'densenet121_m':
        continue
        model = densenet.DenseNet121(16, 45, True)
        # model = torch.load('densenet121_m.pth')
        batch_size = 64

    if torch.cuda.is_available():
        model.cuda()

    alpha = 0.05
    lr = 1e-2
    momentum = 0.9
    max_iteration = 40
    weight_decay = 0.0025
    # Define hyper-parameter

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=True)
    validation_data = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4)

    lambda1 = lambda epoch: 0.9 ** epoch
    lambda2 = lambda epoch: (epoch < 20) * (epoch + 1) + (epoch >= 20)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    total_train_acc = np.zeros(max_iteration)
    total_validation_acc = np.zeros(max_iteration)
    total_train_loss = np.zeros(max_iteration)
    total_validation_loss = np.zeros(max_iteration)

    if __name__ == '__main__':
        for epoch in range(max_iteration):
            total_train_acc[epoch], total_train_loss[epoch] = trainer.train_net(model, train_data, optimizer, epoch,
                                                                                alpha=alpha * lambda2(epoch),
                                                                                metric_learn=True)
            scheduler.step()
            total_validation_acc[epoch], total_validation_loss[epoch] = evaluator.test_net(model, validation_data,
                                                                                           epoch,
                                                                                           alpha=alpha * lambda2(epoch),
                                                                                           metric_learn=True)

        torch.save(model, '{}.pth'.format(chosen_model))

        np.save('Result/{}_train_acc.npy'.format(chosen_model), total_train_acc)
        np.save('Result/{}_validation_acc.npy'.format(chosen_model), total_validation_acc)
        np.save('Result/{}_train_loss.npy'.format(chosen_model), total_train_loss)
        np.save('Result/{}_validation_loss.npy'.format(chosen_model), total_validation_loss)
