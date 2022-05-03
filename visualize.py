import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

path = 'Dataset/test'
model_sequence = [['resnet34', 'resnet34_m'], ['densenet121', 'densenet121_m']]
for i in range(2):
    for j in range(2):
        chosen_model = model_sequence[i][j]
        model = torch.load('{}.pth'.format(chosen_model))
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
                    if chosen_model.endswith('m'):
                        feature, _ = model(image)
                    else:
                        feature = model.draw_feature(image)
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
            if j == 0:
                fig = plt.figure(figsize=(16, 8))
                ax1 = fig.add_subplot(121)
                ax1.scatter(all_feature[:, 0], all_feature[:, 1], c=all_label)
                if i == 0:
                    ax1.set_title('ResNet34')
                else:
                    ax1.set_title('DenseNet121')
            else:
                ax2 = fig.add_subplot(122)
                ax2.scatter(all_feature[:, 0], all_feature[:, 1], c=all_label)
                if i == 0:
                    ax2.set_title('ResNet34M')
                else:
                    ax2.set_title('DenseNet121M')
            del all_feature
            del all_label
    plt.savefig('fig{}.jpg'.format(i + 2))
    plt.close()
