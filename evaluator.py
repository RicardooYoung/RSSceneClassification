import torch
import torch.nn as nn
import time


def test_net(model, validation_data, epoch=0):
    loss_fn = nn.CrossEntropyLoss()
    time_start = time.time()
    validation_loss = 0
    validation_acc = 0
    model.eval()

    for image, label in validation_data:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        out = model(image)
        loss = loss_fn(out, label)

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
