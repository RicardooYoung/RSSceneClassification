import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

path = 'Dataset/test'
# model = torch.load('resnet34_m.pth')
model = torch.load('densenet121.pth')
# model = torch.load('densenet121_m.pth')
if torch.cuda.is_available():
    model.cuda()
test_set = ImageFolder(root=path, transform=ToTensor())
test_data = DataLoader(test_set, batch_size=64, shuffle=False, pin_memory=True)
model.eval()

if __name__ == '__main__':
    with torch.no_grad():
        for image, label in test_data:
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            feature = model.draw_feature(image)
            # feature, _ = model(image)
            if 'all_feature' not in dir():
                all_feature = feature
                all_label = label
            else:
                all_feature = torch.cat((all_feature, feature))
                all_label = torch.cat((all_label, label))

    all_feature = all_feature.cpu()
    all_feature = all_feature.numpy()
    all_label = all_label.cpu()
    all_label = all_label.numpy()
    print('Start Dimensionality Reduction.')
    tsne = TSNE(n_components=2)
    all_feature = tsne.fit_transform(all_feature)
    fig = plt.figure()
    plt.scatter(all_feature[:, 0], all_feature[:, 1], c=all_label)
    # plt.show()
