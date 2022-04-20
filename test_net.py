import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

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

    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        test_acc = 0
        for image, label in test_data:
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            out = model(image)
            _, pred = out.max(1)
            if 'wrong_pred' not in dir():
                wrong_pred = pred[pred != label]
                wrong_label = label[pred != label]
            else:
                wrong_pred = torch.cat((wrong_pred, pred[pred != label]))
                wrong_label = torch.cat((wrong_label, label[pred != label]))
            num_correct = (pred == label).sum().item()
            acc = num_correct / image.shape[0]
            test_acc += acc
        print('{}, Test Acc: {:.6f}.'.format(chosen_model, test_acc / len(test_data)))
        wrong_pred = wrong_pred.cpu().numpy()
        stat = np.zeros(45)
        np.save('Result/{}_wrong_pred.npy'.format(chosen_model), wrong_pred)
        del wrong_pred
        wrong_label = wrong_label.cpu().numpy()
        np.save('Result/{}_wrong_label.npy'.format(chosen_model), wrong_label)
        del wrong_label
