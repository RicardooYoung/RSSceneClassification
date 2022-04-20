import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import LambdaLR
import os

path = 'Dataset/test'
test_set = ImageFolder(root=path, transform=ToTensor())
model_sequence = ['resnet34', 'resnet50', 'densenet121']
for chosen_model in model_sequence:
    model = torch.load('{}.pth'.format(chosen_model))
    if chosen_model == 'resnet34':
        batch_size = 96
    elif chosen_model == 'resnet50':
        batch_size = 32
    elif chosen_model == 'densenet121':
        batch_size = 64

    if torch.cuda.is_available():
        model.cuda()

    test_data = DataLoader(test_set, batch_size=int(batch_size / 2), shuffle=False, num_workers=4,
                           pin_memory=True)
    test_acc = 0
    for image, label in test_data:
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            out = model(image)
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / image.shape[0]
            test_acc += acc
    print('{}: , Test Acc: {:.6f}.'.format(chosen_model, test_acc))
