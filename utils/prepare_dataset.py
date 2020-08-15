import os
import os.path
import sys
from PIL import Image
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader


def get_dataset(name='MNIST',root='../data',transform=transforms.Compose([transforms.ToTensor()]),train=True):
    '''
    name: MNIST,C10,C100, or Folder
    transform:
    '''
    if name=='MNIST':
        dataset=datasets.MNIST(root,train=train,transform=transform,download=True)
        return dataset
    if name=='C10':
        dataset=datasets.CIFAR10(root,train=train,transform=transform,download=True)
        return dataset
    if name=='C100':
        dataset=datasets.CIFAR100(root,train=train,transform=transform,download=True)
        return dataset
