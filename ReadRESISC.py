import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
# from TransCode import dec2bin
from matplotlib import pyplot as plt


class MyDataset(Dataset):
    def __init__(self, path):
        super(MyDataset, self).__init__()
        category = os.listdir(path)
        for i in range(len(category)):
            if category[i].startswith('.'):
                category.pop(i)
                break
        category.sort()
        self.src, self.trg = [], []
        self.transform = transforms.ToTensor()
        for i in range(len(category)):
            current_cate = path + '\\' + category[i]
            file_list = os.listdir(current_cate)
            for j in range(len(file_list)):
                if j >= 100:
                    break
                current_file = current_cate + '\\' + file_list[j]
                self.src.append(current_file)
                self.trg.append(i)
                # self.trg.append(dec2bin(i))

    def __getitem__(self, index):
        return (self.transform(plt.imread(self.src[index]))).cuda(), torch.Tensor(self.trg[index]).cuda()

    def __len__(self):
        return len(self.src)


def read_resisc():
    pass
    # dataset = MyDataset('Data')
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_set, test_set = random_split(dataset, [train_size, test_size])
    # train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    # test_data = DataLoader(test_set, batch_size=64, shuffle=False)
    # return train_data, test_data
