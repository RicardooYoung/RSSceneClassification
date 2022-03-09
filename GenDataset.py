import os
from shutil import copyfile
import random

path = 'Data'
category = os.listdir(path)
for i in range(len(category)):
    if category[i].startswith('.'):
        category.pop(i)
        break

set_path = 'Dataset'
if not os.path.exists(set_path):
    os.mkdir(set_path)

train_path = set_path + '/train'
test_path = set_path + '/test'

if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)

for i in range(len(category)):
    train_path = set_path + '/train/' + category[i]
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    test_path = set_path + '/test/' + category[i]
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    current_cate = path + '/' + category[i]
    file_list = os.listdir(current_cate)
    random.shuffle(file_list)
    for j in range(len(file_list)):
        current_file = current_cate + '/' + file_list[j]
        if j < 560:
            dst_file = train_path + '/' + file_list[j]
        else:
            dst_file = test_path + '/' + file_list[j]
        copyfile(current_file, dst_file)
