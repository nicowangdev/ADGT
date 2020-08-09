from utils.prepare_dataset import get_dataset
from utils.train import normal_train
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ADGT():
    normal_model=None
    gt_model=None

    dataset_name=None
    trainset=None
    testset=None
    trainloader=None
    testloader=None

    use_cuda=False
    def __init__(self,use_cuda=False):
        self.use_cuda=use_cuda
        return
    def prepare_dataset_loader(self,name='MNIST',root='../data',transform=transforms.Compose([transforms.ToTensor()]),
                               train=True,batch_size=128,shuffle=True,num_workers=4):
        self.dataset_name=name
        if train:
            self.trainset=get_dataset(name,root,transform,train)
            self.trainloader=torch.utils.data.DataLoader(self.trainset,batch_size=batch_size,shuffle=shuffle,
                                                         num_workers=num_workers)
        else:
            self.testset=get_dataset(name,root,transform,train)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=shuffle,
                                                           num_workers=num_workers)

    def normal_train(self,model,logdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,):
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()

        logdir=os.path.join(logdir,self.dataset_name,'normal')
        writer = SummaryWriter(log_dir=logdir )
        normal_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda)
        writer.close()
        if self.normal_model is None:
            self.normal_model=model
        print('save model to :',logdir)
        torch.save(model,logdir+'model.ckpt')
        return model




