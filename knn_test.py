import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


def KNN(train_x, train_y, test_x, test_y, k):
    m = test_x.size(0)
    n = train_x.size(0)

    xx = (test_x ** 2).sum(dim=1, keepdim=True).expand(m, n)
    yy = (train_x ** 2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)
    dist_mat = xx + yy - 2 * test_x.matmul(train_x.transpose(0, 1))
    mink_idxs = dist_mat.argsort(dim=-1)
    res = []
    for idxs in mink_idxs:
        res.append(np.bincount(np.array([train_y[idx] for idx in idxs[:k]])).argmax())

    assert len(res) == len(test_y)
    return accuracy_score(test_y, res)


neighbour_path = 'Neighbour'
neighbour_set = ImageFolder(root=neighbour_path, transform=ToTensor())
batch_size = 96
neighbour_data = DataLoader(neighbour_set, batch_size=batch_size, num_workers=4, pin_memory=True)
test_path = 'Dataset/test'
test_set = ImageFolder(root=test_path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=batch_size, num_workers=4, pin_memory=True)
model = torch.load('resnet34_m.pth')
if torch.cuda.is_available():
    model.cuda()

if __name__ == "__main__":
    with torch.no_grad():
        for image, label in neighbour_data:
            if torch.cuda.is_available():
                image = image.cuda()
            feature = model(image)
            feature = feature.cpu()
            if 'all_feature' not in dir():
                all_feature = feature
                all_label = label
            else:
                all_feature = torch.cat((all_feature, feature))
                all_label = torch.cat((all_label, label))
        test_acc = 0
        for image, label in test_data:
            if torch.cuda.is_available():
                image = image.cuda()
            feature = model(image)
            feature = feature.cpu()
            acc = KNN(all_feature, all_label, feature, label, 12)
            test_acc += acc
        print('Test Acc: {:6f}.'.format(test_acc / len(test_data)))
