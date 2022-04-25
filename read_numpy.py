import numpy as np

path = 'Result/resnet34_wrong_pred.npy'
resnet34 = np.load(path)
stat = np.zeros(45)
for i in range(len(stat)):
    stat[i] = len(resnet34[resnet34 == i])
stat = np.argsort(stat)
