import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # 获取当前目录
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir)) # 获取上级目录

import sys
sys.path.append(parent_dir_path) # 添加上级目录

import d2l

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True
)
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=d2l.get_fasion_mnnnist_labels(y))
