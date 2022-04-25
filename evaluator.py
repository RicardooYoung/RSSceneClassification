import torch
import torch.nn as nn
import time

from triplet_loss import TripletLoss


def test_net(model, validation_data, epoch=0, alpha=0, metric_learn=False):
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = TripletLoss()
    time_start = time.time()
    validation_loss = 0
    validation_acc = 0
    model.eval()

    with torch.no_grad():
        for image, label in validation_data:
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            if metric_learn:
                feature, out = model(image)
                loss = loss_fn1(out, label) + alpha * loss_fn2(feature, label)
            else:
                out = model(image)
                loss = loss_fn1(out, label)

            validation_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / image.shape[0]
            validation_acc += acc

        time_end = time.time()

        print(
            'Epoch: {}, Test Loss: {:.6f}, Test Acc: {:.6f}, Time Elapsed: {:.3f}s'
                .format(epoch + 1, validation_loss / len(validation_data), validation_acc / len(validation_data),
                        time_end - time_start))

    return validation_acc / len(validation_data), validation_loss / len(validation_data)
