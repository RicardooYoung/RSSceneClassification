import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import LambdaLR
import os

import resnet
import densenet
import trainer
import evaluator

train_path = 'Dataset/train'
validation_path = 'Dataset/validation'
train_set = ImageFolder(root=train_path, transform=ToTensor())
validation_set = ImageFolder(root=validation_path, transform=ToTensor())

model_sequence = ['resnet34', 'resnet50', 'densenet121', 'resnet34_m']

if not os.path.exists('Result'):
    os.mkdir('Result')

for chosen_model in model_sequence:
    if chosen_model == 'resnet34':
        model = resnet.PreResNet34(45)
        batch_size = 96
        alpha = 0
        metric_learn = False
    elif chosen_model == 'resnet50':
        model = resnet.PreResNet50(45)
        batch_size = 32
        alpha = 0
        metric_learn = False
    elif chosen_model == 'densenet121':
        model = densenet.DenseNet121(16, 45)
        batch_size = 64
        alpha = 0
        metric_learn = False
    elif chosen_model == 'resnet34_m':
        model = resnet.PreResNet34(45, True)
        batch_size = 96
        alpha = 0.1
        metric_learn = True

    if torch.cuda.is_available():
        model.cuda()

    lr = 1e-2
    momentum = 0.9
    max_iteration = 30
    weight_decay = 1e-4
    # Define hyper-parameter

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=True)
    validation_data = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4)

    lambda1 = lambda epoch: 0.9 ** epoch
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    total_train_acc = np.zeros(max_iteration)
    total_validation_acc = np.zeros(max_iteration)
    total_train_loss = np.zeros(max_iteration)
    total_validation_loss = np.zeros(max_iteration)

    if __name__ == '__main__':
        for epoch in range(max_iteration):
            total_train_acc[epoch], total_train_loss[epoch] = trainer.train_net(model, train_data, optimizer, epoch,
                                                                                alpha=alpha, metric_learn=metric_learn)
            scheduler.step()
            total_validation_acc[epoch], total_validation_loss[epoch] = evaluator.test_net(model, validation_data,
                                                                                           epoch, alpha=alpha,
                                                                                           metric_learn=metric_learn)

    torch.save(model, '{}.pth'.format(chosen_model))

    np.save('Result/{}_train_acc.npy'.format(chosen_model), total_train_acc)
    np.save('Result/{}_validation_acc.npy'.format(chosen_model), total_validation_acc)
    np.save('Result/{}_train_loss.npy'.format(chosen_model), total_train_loss)
    np.save('Result/{}_validation_loss.npy'.format(chosen_model), total_validation_loss)

# epoch = np.linspace(1, max_iteration, max_iteration)
# new_tick = np.linspace(0, max_iteration, max_iteration + 1)
# plt.figure()
# plt.plot(epoch, total_train_acc[0:max_iteration], '-.^')
# plt.plot(epoch, total_validation_acc[0:max_iteration], '-.^')
# plt.title('Train & Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.xticks(new_tick)
# plt.legend(['Train Acc', 'Test Acc'])
# plt.savefig('fig1.jpg')
# plt.show()
#
# plt.figure()
# plt.plot(epoch, total_train_loss, '-.^')
# plt.plot(epoch, total_validation_loss, '-.^')
# plt.title('Train & Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.xticks(new_tick)
# plt.legend(['Train Loss', 'Test Loss'])
# plt.savefig('fig2.jpg')
# plt.show()
