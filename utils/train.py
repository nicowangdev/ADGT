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

def normal_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda):

    for i in range(max_epoch):
        index=0
        lossmean = 0
        num = 0
        for data,label in trainloader:
            index+=1
            optimizer.zero_grad()
            model.train()
            if use_cuda:
                data,label=data.cuda(),label.cuda()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if index%10==0:
                print(i, index, loss.item())
                lossmean += loss.item()
                num += 1
        writer.add_scalar('loss', lossmean / num, i)
        acc=get_acc(testloader,model,use_cuda)
        writer.add_scalar('acc', acc.item(), i)
        print(i, acc.item())
        if schedule is not None:
            schedule.step()
    return model

def clip(x,min,max):
    return max-F.relu(max-min-F.relu(x-min))

def attack_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,attack,min,max):

    if use_cuda:
        attack,min,max=attack.cuda(),min.cuda(),max.cuda()
    for i in range(max_epoch):
        index=0
        lossmean = 0
        num = 0
        for data,label in trainloader:
            index+=1
            optimizer.zero_grad()
            model.train()
            if use_cuda:
                data,label=data.cuda(),label.cuda()
            attack_temp = attack[label]
            data = clip(data+attack_temp,min,max)
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if index%10==0:
                print(i, index, loss.item())
                lossmean += loss.item()
                num += 1
        writer.add_scalar('loss', lossmean / num, i)
        acc=get_acc(testloader,model,use_cuda)
        writer.add_scalar('acc', acc.item(), i)
        print(i, acc.item())
        if schedule is not None:
            schedule.step()
    return model

def removeSPP_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,mu,sigma,prob):

    if use_cuda:
        mu,sigma=mu.cuda(),sigma.cuda()
    for i in range(max_epoch):
        index=0
        lossmean = 0
        num = 0
        for data,label in trainloader:
            index+=1
            optimizer.zero_grad()
            model.train()
            if use_cuda:
                data,label=data.cuda(),label.cuda()
            mu_temp,sigma_temp=mu[label],sigma[label]
            R=torch.randn_like(data)*sigma_temp+mu_temp
            P=torch.bernoulli(torch.ones_like(data)*prob)
            data=data*(1-P)+R*P
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if index%10==0:
                print(i, index, loss.item())
                lossmean += loss.item()
                num += 1
        writer.add_scalar('loss', lossmean / num, i)
        acc=get_acc(testloader,model,use_cuda)
        writer.add_scalar('acc', acc.item(), i)
        print(i, acc.item())
        if schedule is not None:
            schedule.step()
    return model

def get_acc(loader,model,use_cuda):
    right=0
    all=0
    model.eval()
    for data,label in loader:
        if use_cuda:
            data,label=data.cuda(),label.cuda()
        out=model(data)
        pre = torch.argmax(out, 1)
        right+=torch.sum((pre==label).float())
        all+=data.size(0)
    return right/all

