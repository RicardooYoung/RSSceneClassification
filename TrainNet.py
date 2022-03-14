import torch
import torch.nn as nn
import time


def train_net(model, train_data, epoch, lr=1e-1, momentum=0.0):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print('Epoch {} start.'.format(epoch + 1))

    time_start = time.time()
    train_loss = 0
    train_acc = 0
    model.train()

    for image, label in train_data:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        out = model(image)
        loss = loss_fn(out, label)

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
        'Epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Time elapsed: {:.3f}s'
            .format(epoch + 1, train_loss / len(train_data), train_acc / len(train_data), time_end - time_start))

    return train_acc / len(train_data)
