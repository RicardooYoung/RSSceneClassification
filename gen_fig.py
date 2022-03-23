import os
import math
from matplotlib import pyplot as plt

path = 'Data'
category = os.listdir(path)
for i in range(len(category)):
    if category[i].startswith('.'):
        category.pop(i)
        break
plt.figure(figsize=(12, 20))
for i in range(len(category)):
    category_path = path + '/' + category[i]
    fig_list = os.listdir(category_path)
    fig_path = category_path + '/' + fig_list[0]
    row = math.floor(i / 5)
    col = i - row * 5
    img = plt.imread(fig_path)
    plt.subplot(9, 5, i + 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    img_title = category[i]
    img_title = img_title.replace('_', ' ')
    plt.title(img_title)

plt.savefig('fig3.jpg')
plt.show()
