import torch
import torch.nn as nn
import numpy as np


class NCALoss(nn.Module):
    def __init__(self):
        super(NCALoss, self).__init__()
        self.norm = 2

    def forward(self, feature, label, proxy):
        positive_proxy = torch.ones(feature.size(0), 512, device='cuda', requires_grad=False)
        pdist = nn.PairwiseDistance(p=self.norm)
        denominator = 0
        for i in range(label.size(0)):
            positive_proxy[i] = proxy[label[i]]
        idx = np.zeros(label.size(0))
        for j in range(44):
            negative_proxy = torch.ones_like(positive_proxy, requires_grad=False)
            for i in range(label.size(0)):
                if idx[i] == label[i]:
                    idx[i] += 1
                negative_proxy[i] = proxy[int(idx[i])]
            denominator += torch.exp(-pdist(feature, negative_proxy))
            idx += 1
        numerator = torch.exp(-pdist(feature, positive_proxy))
        loss = -torch.log(torch.div(numerator, denominator))
        loss = loss.sum()
        return loss
