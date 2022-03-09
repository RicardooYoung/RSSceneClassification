from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch

path = 'Data'
dataset = ImageFolder(root=path, transform=ToTensor)
