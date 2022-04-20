from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = 'Dataset/test'
model = torch.load('resnet34_m.pth')
test_set = ImageFolder(root=path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
model.eval()

if __name__ == '__main__':
    with torch.no_grad():
        for image, _ in test_data:
            feature = model(image)
            if 'all_feature' not in dir():
                all_feature = feature
            else:
                all_feature = torch.cat((all_feature, feature))

    all_feature = all_feature.numpy()
    all_feature = all_feature.T
    pca = PCA(n_components=3)
    pca.fit_transform(all_feature)
    ax = plt.figure()
    ax.scatter(pca.components_[0, :], pca.components_[1, :], pca.components_[2, :])
