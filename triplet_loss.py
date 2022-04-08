import torch.nn as nn
from torch.nn import functional as f


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negative):
        positive_dist = (anchor - positive).pow(2).sim(1)
        negative_dist = (anchor - negative).pow(2).sim(1)
        loss = f.relu(positive_dist - negative_dist + self.margin)
        return loss.mean()
