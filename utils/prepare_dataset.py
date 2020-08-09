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

ROOT='../data'

def get_train_dataset(name='MNIST',transform=transforms.ToTensor(),aug=True):
    if name=='MNIST':
        dataset=datasets.MNIST(ROOT,train=True,transform=transform,download=True)
        return dataset
    if name=='C10':
        if aug:
            transform = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([ transforms.ToTensor()])
        dataset=datasets.CIFAR10(ROOT,train=True,transform=transform,download=True)

        return dataset

def get_test_dataset(name='MNIST',transform=transforms.ToTensor(),aug=True):
    if name=='MNIST':
        dataset=datasets.MNIST(ROOT,train=False,transform=transform,download=True)
        return dataset
    if name=='C10':
        if not aug:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset=datasets.CIFAR10(ROOT,train=False,transform=transform,download=True)
        return dataset