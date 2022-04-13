import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import resnet
import triplet_loss

batch_size = 64

train_path = 'Dataset/train'
train_set = ImageFolder(root=train_path, transform=ToTensor())
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

model = resnet.ResNet34(45)

if torch.cuda.is_available():
    model.cuda()

max_iteration = 20
lr = 1e-2
momentum = 0.9
margin = 0.2

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

if __name__ == '__main__':
    for epoch in range(max_iteration):
        triplet_loss.triplet_loss(model, train_data, optimizer, margin, epoch)
