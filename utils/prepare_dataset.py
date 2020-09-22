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
import urllib.request

import zipfile

def download_progress(blocknum, blocksize, totalsize):

    """

    Show download progress

    :param blocknum:

    :param blocksize:

    :param totalsize:

    :return:

    """

    readsofar = blocknum * blocksize

    if totalsize > 0:

        percent = readsofar * 1e2 / totalsize

        s = "\r%5.1f%% %*d / %d" % (

            percent, len(str(totalsize)), readsofar, totalsize)

        sys.stderr.write(s)

        if readsofar >= totalsize: # near the end

            sys.stderr.write("\n")

    else: # total size is unknown

        sys.stderr.write("read %d\n" % (readsofar,))

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
    if name=='Flower102':
        data_dir = os.path.join(root,'flower_data')
        print(data_dir)
        train_dir = os.path.join(data_dir, 'train')

        valid_dir = os.path.join(data_dir, 'valid')

        # Download the dataset""

        if not os.path.exists(data_dir):
            print("Downloading the dataset...")
            zip_file_name = os.path.join(root, "flower_data.zip")

            urllib.request.urlretrieve(
                "https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip",

                zip_file_name, download_progress)

            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(root)

            #os.remove(zip_file_name)
        dirs = {'train': train_dir,'valid': valid_dir}
        if train:
            x='train'
        else:
            x='valid'
        dataset=datasets.ImageFolder(dirs[x], transform=transform)
        return dataset
    if name=='RestrictedImageNet':
        if train:
            dir=os.path.join(root,"RestrictedImageNet/train")
        else:
            dir = os.path.join(root, "RestrictedImageNet/val")
        print(dir)
        dataset=datasets.ImageFolder(dir,transform=transform)
        return dataset
