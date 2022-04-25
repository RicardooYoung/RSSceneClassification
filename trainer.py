import torch
import torch.nn as nn
import time
from tqdm import tqdm

from triplet_loss import TripletLoss


def train_net(model, train_data, optimizer, epoch=0, alpha=0, metric_learn=False):
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = TripletLoss()

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
            .format(epoch + 1, train_loss / len(train_data), train_acc / len(train_data), time_end - time_start))

    return train_acc / len(train_data), train_loss / len(train_data)
