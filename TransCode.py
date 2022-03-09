import numpy as np
import torch


def dec2bin(num):
    is_int = False
    if isinstance(num, int):
        num = np.array([num])
        is_int = True
        bi_res = np.zeros(6)
    else:
        bi_res = np.zeros((len(num), 6))
    for i in range(len(num)):
        while num[i] > 0:
            for j in range(6):
                if num[i] >= pow(2, 5 - j):
                    num[i] = num[i] - pow(2, 5 - j)
                    if is_int:
                        bi_res[j] = 1
                    else:
                        bi_res[i][j] = 1
    return bi_res


def bin2dec(code):
    num = np.zeros(len(code))
    for i in range(len(code)):
        for j in range(6):
            num[i] = num[i] + code[i][j] * pow(2, 5 - j)
    return num
