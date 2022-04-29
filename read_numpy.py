import os

import matplotlib.pyplot as plt
import numpy as np

model_sequence = ['resnet34', 'resnet34_m', 'densenet121', 'densenet121_m']
fig = plt.figure()
max_iteration = 40
epoch = np.linspace(1, max_iteration, max_iteration)
for chosen_model in model_sequence:
    path = 'Result/{}_validation_acc.npy'.format(chosen_model)
    acc = np.load(path)
    plt.plot(epoch, acc, '-.')
    plt.title('Learning Curve on Validation Set')

plt.legend(['ResNet34', 'ResNet34M', 'DenseNet121', 'DenseNet121M'])
plt.savefig('fig1.jpg')
plt.close()

for chosen_model in model_sequence:
    with open('{}_result.txt'.format(chosen_model), 'w') as f:

        path = 'Result/{}_wrong_label.npy'.format(chosen_model)
        wrong_label = np.load(path)
        path = 'Result/{}_wrong_pred.npy'.format(chosen_model)
        wrong_pred = np.load(path)
        f.write('{}: \n'.format(chosen_model))
        category = os.listdir('Dataset/test')
        total_count = 140
        for i in range(len(category)):
            if category[i].startswith('.'):
                category.pop(i)
                break
        for i in range(len(category)):
            wrong_category = np.zeros(len(category))
            count = len(wrong_label[wrong_label == i])
            f.write('{}: {:3f}, '.format(category[i], 1 - count / total_count))
            for j in range(len(wrong_label)):
                if wrong_label[j] == i:
                    wrong_category[wrong_pred[j]] += 1
            wrong_frequency = np.argsort(wrong_category)
            for j in range(len(wrong_frequency)):
                temp = wrong_frequency[-(j + 1)]
                if wrong_category[temp] == 0:
                    break
                f.write('{}: {}. '.format(category[temp], wrong_category[temp]))
            f.write(' \n\n')
