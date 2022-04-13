from torch.utils.data import Dataset
import os


class Triplet(Dataset):
    def __init__(self):
        super(Triplet, self).__init__()
        self.image = []
        path = 'Dataset/train'
        category = os.listdir(path)
        self.category_count = 0
        self.image_count = 0
        for i in range(len(category)):
            if category[i].startswith('.'):
                continue
            category_path = os.path.join(path, category[i])
            file_list = os.listdir(category_path)
            self.category_count += 1
            for j in range(len(file_list)):
                if file_list[j].startswith('.'):
                    continue
                file_path = os.path.join(category_path, file_list[j])
                self.image.append(file_path)
                if i == 0:
                    self.image_count += 1

    def __getitem__(self, index):
        pass
