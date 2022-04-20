import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time

import resnet


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = torch.addmm(1, dist, -2, inputs, inputs.t())
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


model = resnet.PreResNet34(45, True)
model.cuda()
train_path = 'Dataset/train'
train_set = ImageFolder(root=train_path, transform=ToTensor())
batch_size = 64
lr = 1e-2
momentum = 0.9
weight_decay = 1e-4
max_iteration = 30
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)
# loss_fn = TripletMarginLoss(margin=0.3)
loss_fn = TripletLoss()
model.train()
if __name__ == '__main__':
    for epoch in range(max_iteration):
        time_start = time.time()
        for image, label in train_data:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            feature = model(image)
            loss = loss_fn(feature, label)
            loss.backward()
        time_end = time.time()
        print('Epoch {}, Time Elapsed: {:.3f}s'.format(epoch + 1, time_end - time_start))

torch.save(model, 'resnet34_m.pth')
