import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import evaluator

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

    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,
                           pin_memory=True)
    evaluator.test_net(model, test_data)
