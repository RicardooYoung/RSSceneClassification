import torch
import torch.nn as nn
import time


def test_net(model, test_data, epoch=0):
    loss_fn = nn.CrossEntropyLoss()
    time_start = time.time()
    test_loss = 0
    test_acc = 0
    model.eval()

    for image, label in test_data:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        out = model(image)
        loss = loss_fn(out, label)

        test_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / image.shape[0]
        test_acc += acc

    time_end = time.time()

    print(
        'Epoch: {}, Test Loss: {:.6f}, Test Acc: {:.6f}, Time Elapsed: {:.3f}s'
            .format(epoch + 1, test_loss / len(test_data), test_acc / len(test_data), time_end - time_start))

    return test_acc / len(test_data), test_loss / len(test_data)
