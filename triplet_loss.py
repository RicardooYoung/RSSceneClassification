import torch.nn as nn
from torch.nn import functional as f


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negetive):
        positive_dist = (anchor - positive).pow(2).sim(1)
        negetive_dist = (anchor - negetive).pow(2).sim(1)
        loss = f.relu(positive_dist - negetive_dist + self.margin)
        return loss.mean()
