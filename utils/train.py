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
                     schedule,criterion,max_epoch,writer,use_cuda,img=None,target=None,method=None
              ,explain=None,explain_dir=None):
    b=0.2
    for i in range(max_epoch):
        if i %5==0:
            if img is not None:
                pth = os.path.join(explain_dir, str(i))
                for m in method:
                    print(m)
                    temp_model = model
                    if m == 'IntegratedGradients':
                        temp_model = torch.nn.DataParallel(model)
                    temp_model.eval()
                    explain(img, target, pth, method=m, model=temp_model, random=False, attack=False)
        index=0
        lossmean = 0
        accmean=0
        num = 0
        for data,label in trainloader:
            index+=1
            optimizer.zero_grad()
            model.train()
            if use_cuda:
                data,label=data.cuda(),label.cuda()
            out = model(data)
            loss = criterion(out, label)
            #loss=torch.abs(criterion(out, label)-b)+b
            loss.backward()
            optimizer.step()

            if index % 10 == 0:
                print(i, index, loss.item())
                lossmean += loss.item()
                pre = torch.argmax(out, 1)
                accmean += torch.mean((pre == label).float()).item()
                num += 1
        writer.add_scalar('train_loss', lossmean / num, i)
        writer.add_scalar('train_acc', accmean / num, i)
        acc = get_acc(testloader, model, use_cuda)
        writer.add_scalar('test_acc', acc.item(), i)
        print(i, accmean / num, acc.item())
        if schedule is not None:
            schedule.step()
    return model
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
import random
def RPB_train(model,rpb,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,prob=1,alpha=0.1,img=None,target=None,method=None
              ,explain=None,explain_dir=None):
    if use_cuda:
        rpb=rpb.cuda()
    rpb.train()
    #label_smoothing=LabelSmoothing(prob)
    for i in range(max_epoch):
        if i %5==0:
            if img is not None:
                pth = os.path.join(explain_dir, str(i))
                for m in method:
                    print(m)
                    temp_model = model
                    if m == 'IntegratedGradients':
                        temp_model = torch.nn.DataParallel(model)
                    temp_model.eval()
                    explain(img, target, pth, method=m, model=temp_model, random=False, attack=False)
        index = 0
        lossmean = 0
        accmean = 0
        num = 0
        for data,label in trainloader:
            index+=1
            optimizer.zero_grad()
            model.train()
            if use_cuda:
                data,label=data.cuda(),label.cuda()
            if random.random()<prob:
                data=rpb(data)
            out = model(data)
            loss = criterion(out, label)
            #loss=label_smoothing(out,label)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print(i, index, loss.item())
                lossmean += loss.item()
                pre = torch.argmax(out, 1)
                accmean += torch.mean((pre == label).float()).item()
                num += 1
        writer.add_scalar('train_loss', lossmean / num, i)
        writer.add_scalar('train_acc', accmean / num, i)
        acc = get_acc(testloader, model, use_cuda)
        writer.add_scalar('test_acc', acc.item(), i)
        print(i, accmean / num, acc.item())
        if schedule is not None:
            schedule.step()
    return model

def clip(x,min,max):
    return max-F.relu(max-min-F.relu(x-min))

def attack_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,attack,min,max,img,target,method,explain,explain_dir):

    if use_cuda:
        attack,min,max=attack.cuda(),min.cuda(),max.cuda()

    for i in range(max_epoch):
        if i %5==0:
            if img is not None:
                pth = os.path.join(explain_dir, str(i))
                for m in method:
                    print(m)
                    temp_model = model
                    if m == 'IntegratedGradients':
                        temp_model = torch.nn.DataParallel(model)
                    temp_model.eval()
                    explain(img, target, pth, method=m, model=temp_model, random=False, attack=True)
        index=0
        lossmean = 0
        accmean=0
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
            if index % 10 == 0:
                print(i, index, loss.item())
                lossmean += loss.item()
                pre = torch.argmax(out, 1)
                accmean += torch.mean((pre == label).float()).item()
                num += 1
        writer.add_scalar('train_loss', lossmean / num, i)
        writer.add_scalar('train_acc', accmean / num, i)
        acc = get_acc(testloader, model, use_cuda)
        writer.add_scalar('test_acc', acc.item(), i)
        print(i, accmean/num,acc.item())
        if schedule is not None:
            schedule.step()

    return model

def removeSPP_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,mu,sigma,prob,max=None,min=None):

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
            if max is not None:
                if use_cuda:
                    min, max = min.cuda(),max.cuda()
                R=clip(R,min,max)
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
def remove_attack_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,mu,sigma,prob,attack,max=None,min=None):

    if use_cuda:
        mu,sigma=mu.cuda(),sigma.cuda()
        attack, min, max = attack.cuda(), min.cuda(), max.cuda()
    pan_attack = 1-torch.sign(torch.sum(torch.abs(attack), 0, keepdim=True))
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
            if max is not None:
                if use_cuda:
                    min, max = min.cuda(),max.cuda()
                R=clip(R,min,max)
            P=torch.bernoulli(torch.ones_like(data)*prob)*pan_attack
            data=data*(1-P)+R*P

            #data=data*prob
            attack_temp = attack[label]
            data = clip(data + attack_temp, min, max)
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

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,alpha,use_cuda):

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
            inputs, targets_a, targets_b, lam = mixup_data(data, label, alpha, use_cuda)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
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

from utils.fast_gradient_sign_untargeted import FastGradientSignUntargeted

def adversarial_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,min,max,perturbation_type='l2',eps=0.3):
    min,max=torch.min(min),torch.max(max)
    attack= FastGradientSignUntargeted(model, eps, 2/255, min_val=min, max_val=max, max_iters=10,
                                        _type=perturbation_type)
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
            adv_data=attack.perturb(data, label, 'mean', True)

            out = model(adv_data)
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
def L1_train(model,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,alpha=0.1):

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
            data.requires_grad_(True)
            out = model(data)
            iii = torch.LongTensor(range(data.size(0))).cuda()
            L1=torch.abs(compute_grad(out[iii,label],data)).view(data.size(0),-1).mean()
            loss0=criterion(out, label)
            loss = loss0+alpha*L1
            loss.backward()
            optimizer.step()
            if index%10==0:
                print(i, index, loss.item(),L1.item())
                lossmean += loss.item()
                num += 1
        writer.add_scalar('loss', lossmean / num, i)
        acc=get_acc(testloader,model,use_cuda)
        writer.add_scalar('acc', acc.item(), i)
        print(i, acc.item())
        if schedule is not None:
            schedule.step()
    return model
def L1_RPB_train(model,rpb,trainloader,testloader,optimizer,
                     schedule,criterion,max_epoch,writer,use_cuda,prob,alpha):
    if use_cuda:
        rpb=rpb.cuda()
    rpb.train()
    #label_smoothing=LabelSmoothing(prob)
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
            data=rpb(data)
            data.requires_grad_(True)
            out = model(data)
            iii = torch.LongTensor(range(data.size(0))).cuda()
            L1 = torch.abs(compute_grad(out[iii, label], data)).view(data.size(0), -1).mean()
            loss0 = criterion(out, label)
            loss = loss0 + alpha * L1

            loss.backward()
            optimizer.step()
            if index%10==0:
                print(i, index, loss.item(),L1.item())
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

def compute_grad(d_out, x_in):
  batch_size = x_in.size(0)
  grad_dout = torch.autograd.grad(
    outputs=d_out.sum(), inputs=x_in,
    create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  #grad_dout2 = grad_dout.pow(2)
  #assert (grad_dout2.size() == x_in.size())
  #reg = grad_dout2.view(batch_size, -1).mean(1)
  return grad_dout

