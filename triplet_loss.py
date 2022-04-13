import torch
import torch.nn as nn
import torch.nn.functional as f
import time
from tqdm import tqdm


def calculate_dist(anchor, positive, negative, margin=0.2):
    dist = f.pairwise_distance(anchor, positive) - f.pairwise_distance(anchor, negative) + margin
    if dist <= 0:
        return False
    else:
        return True


def triplet_loss(model, train_data, optimizer, margin=0.2, epoch=0):
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.TripletMarginLoss(margin)

    print('Epoch {} start.'.format(epoch + 1))

    time_start = time.time()
    count = 0
    model.train()

    for image, label in tqdm(train_data):
        if torch.cuda.is_available():
            image = image.cuda()
        out = model(image)
        for i in range(len(label) - 2):
            for j in range(len(label) - i - 1):
                for k in range(len(label) - i - j - 2):
                    if label[i] == label[i + j + 1]:
                        if label[i] != label[i] != label[i + j + k + 2]:
                            calculate_dist(out[i], out[j], out[k], margin)
                            if 'anchor' not in dir():
                                anchor = out[i]
                                positive = out[j]
                                negative = out[k]
                            else:
                                anchor = torch.stack((anchor, out[i]), dim=0)
                                positive = torch.stack((positive, out[j]), dim=0)
                                negative = torch.stack((negative, out[k]), dim=0)
        if 'anchor' not in dir():
            continue
        loss = loss_fn(anchor, positive, negative)
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_end = time.time()

    print('Epoch: {}, {} mini-batches have valid triplets'.format(epoch, count))
