import os
from shutil import copyfile
import random


def gen_dataset():
    path = 'Data'
    category = os.listdir(path)
    for i in range(len(category)):
        if category[i].startswith('.'):
            category.pop(i)
            break

    set_path = 'Dataset'
    if not os.path.exists(set_path):
        os.mkdir(set_path)

    train_path = os.path.join(set_path, 'train')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    validation_path = os.path.join(set_path, 'validation')
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)
    test_path = os.path.join(set_path, 'test')
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    for i in range(len(category)):
        train_path = set_path + '/train/' + category[i]
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        validation_path = set_path + '/validation/' + category[i]
        if not os.path.exists(validation_path):
            os.mkdir(validation_path)
        test_path = set_path + '/test/' + category[i]
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        current_cate = path + '/' + category[i]
        file_list = os.listdir(current_cate)
        random.shuffle(file_list)
        for j in range(len(file_list)):
            current_file = current_cate + '/' + file_list[j]
            if j < 140:
                dst_file = train_path + '/' + file_list[j]
            elif 140 <= j < 210:
                dst_file = validation_path + '/' + file_list[j]
            else:
                dst_file = test_path + '/' + file_list[j]
            copyfile(current_file, dst_file)
        print('Class {} created.'.format(category[i]))

    print('Dataset successfully generated.')


gen_dataset()
