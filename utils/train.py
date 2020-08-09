from utils import prepare_dataset
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from model import resnet

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
dataset='MNIST'#MNIST,C10
model='linear'
attack_state=1
para_mode='s'
scale=1.0
MAXITER=200
ROOT='./result/'

import math
def sgdr(period, batch_idx):
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx / restart_period > 1.:
        batch_idx = batch_idx - restart_period
    restart_period = restart_period * 2.
    radians = math.pi * (batch_idx / restart_period)
    return 0.5 * (1.0 + math.cos(radians))
def train(model,trainloader,testloader,datasetname='C10',attack_state=2,scale=1):
    if datasetname=='MNIST':
        attack=torch.zeros(10,1,28,28).cuda()
    else:
        attack = torch.zeros(10, 3, 32, 32).cuda()
    for i in range(10):
        attack[i,:,i*2,0]=2
        if attack_state==2:
            for j in range(10):
                attack[i,:,j*2,0]-=1
    if para_mode=='s':
        attack*=scale
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
    #optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    #acc = get_acc(testloader, model)
    #writer.add_scalar('acc', acc.item(), 0)

    for i in range(MAXITER):
        index = 0
        if i==150:
            for p in optimizer.param_groups:
                p['lr'] = 0.01
        elif i==225:
            for p in optimizer.param_groups:
                p['lr'] = 0.001

        lossmean = 0
        num = 0
        for data,label in trainloader:
            iii = torch.LongTensor(range(data.size(0))).cuda()
            index+=1
            optimizer.zero_grad()
            model.train()
            #print(data)
            data,label=data.cuda(),label.cuda()
            if attack_state!=0:
                attack_temp=attack[label]
                if para_mode=='p':
                    pr=torch.ones(data.size(0))*scale
                    r=torch.bernoulli(pr).cuda()
                    attack_temp=attack_temp*r.view(-1,1,1,1)
                data=F.relu(1-F.relu(1-(data+attack_temp)))
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if index%10==0:
                print(i, index, loss.item())
                lossmean += loss.item()
                num += 1
        writer_raw.add_scalar('loss', lossmean / num, i)
        acc=get_acc(testloader,model)
        writer_raw.add_scalar('acc', acc.item(), i)
        print(i, acc.item())
    return model

def compute_grad(d_out, x_in):
  grad_dout = torch.autograd.grad(
    outputs=d_out.sum(), inputs=x_in,
    create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  return grad_dout


def get_acc(loader,model):
    right=0
    all=0
    model.eval()
    for data,label in loader:
        data,label=data.cuda(),label.cuda()
        out=model(data)
        if isinstance(out,tuple):
            out=out[0]
        pre = torch.argmax(out, 1)
        right+=torch.sum((pre==label).float())
        all+=data.size(0)
    return right/all

class Final(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.fc=nn.Linear(in_channels, num_classes,bias=True)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Conv' or 'SNConv') != -1:
            m.weight.data.normal_(0.0, 0.02)
    def forward(self, x):
        out=self.fc(x.view(x.size(0),-1))
        return out


writer_raw = SummaryWriter( log_dir=ROOT+dataset+'/'+str(attack_state)+'/'+model+para_mode+'_'+str(scale))

print('prepare_dataset:'+dataset)
print('attack_state:'+str(attack_state))
train_set=prepare_dataset.get_train_dataset(name=dataset,aug=False)
test_set=prepare_dataset.get_test_dataset(name=dataset,aug=False)
train_loader=torch.utils.data.DataLoader(train_set,batch_size=8,shuffle=True,num_workers=4)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=128,shuffle=False,num_workers=4)

if dataset=='MNIST':
    if model=='linear':
        net=Final(784,10).cuda()
    else:
        net=resnet.resnet18(indim=1).cuda()
else:
    net = resnet.resnet18(indim=3).cuda()
print(net)
print('training start!')
train(net,train_loader,test_loader,datasetname=dataset,attack_state=attack_state,scale=scale)
writer_raw.close()
#torch.save(net,dataset+str(attack_state)+'prenet.ckpt')

