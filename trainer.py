import torch
import torch.nn as nn
import time
from tqdm import tqdm
import numpy as np

from triplet_loss import TripletLoss
from nca import NCALoss


def train_net(model, train_data, optimizer, epoch=0, alpha=0, metric_learn=False, proxy=None):
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = TripletLoss()
    loss_fn3 = NCALoss()

    print('Epoch {} start.'.format(epoch + 1))

    time_start = time.time()
    train_loss = 0
    train_acc = 0
    model.train()

    for image, label in tqdm(train_data):
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        if metric_learn:
            feature, out = model(image)
            loss = loss_fn1(out, label) + alpha * loss_fn2(feature, label)
        elif proxy is not None:
            loss = torch.zeros(1, device='cuda')
            triplet = nn.TripletMarginLoss(margin=0.5, p=2)
            positive_proxy = torch.ones(image.size(0), 512, device='cuda', requires_grad=False)
            feature, out = model(image)
            for i in range(label.size(0)):
                positive_proxy[i] = proxy[label[i]]
            idx = np.zeros(label.size(0))
            for j in range(44):
                negative_proxy = torch.ones_like(positive_proxy, device='cuda', requires_grad=False)
                for i in range(label.size(0)):
                    if idx[i] == label[i]:
                        idx[i] += 1
                    negative_proxy[i] = proxy[int(idx[i])]
                idx += 1
                temp_loss = triplet(feature, positive_proxy, negative_proxy)
                loss += temp_loss

            # feature, out = model(image)
            # loss = loss_fn3(feature, label, proxy)

        else:
            out = model(image)
            loss = loss_fn1(out, label)

        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / image.shape[0]
        train_acc += acc

    time_end = time.time()

    print(
        'Epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Time Elapsed: {:.3f}s'
        .format(epoch + 1, train_loss.item() / len(train_data), train_acc / len(train_data), time_end - time_start))

    return train_acc / len(train_data), train_loss.item() / len(train_data)
