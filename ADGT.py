from utils.prepare_dataset import get_dataset
from utils.train import normal_train,attack_train,clip,removeSPP_train,remove_attack_train,adversarial_train,\
    RPB_train,mixup_train,L1_train,L1_RPB_train
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils.visualization import save_images
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class ADGT():
    normal_model=None
    gt_model=None
    improve_model=None
    RPB_model=None

    dataset_name=None
    trainset=None
    testset=None
    trainloader=None
    testloader=None

    attack=None
    min=None
    max=None
    mu=None
    use_cuda=False
    aug=False

    nclass={'MNIST':10,'C10':10,'C100':100,'Flower102':102,'RestrictedImageNet':9}

    def __init__(self,name='MNIST',nclass=None,use_cuda=False,min=None,max=None,attack=None,normal_model=None,
                 gt_model=None,aug=False):
        self.aug=aug
        self.use_cuda=use_cuda
        self.dataset_name=name
        if nclass is not None:
            self.nclass[name]=nclass
        self.min=min
        self.max=max
        self.attack=attack
        self.normal_model=normal_model
        self.gt_model=gt_model
        return
    def save_gt(self,checkpointdir):
        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(self.gt_model,os.path.join(checkpointdir,'model.ckpt'))
        np.save(os.path.join(checkpointdir,'min.npy'),self.min.numpy())
        np.save(os.path.join(checkpointdir, 'max.npy'), self.max.numpy())
        np.save(os.path.join(checkpointdir, 'attack.npy'), self.attack.numpy())
    def load_gt(self,checkpointdir):
        model=torch.load(os.path.join(checkpointdir,'model.ckpt'))
        min=np.load(os.path.join(checkpointdir,'min.npy'))
        max=np.load(os.path.join(checkpointdir,'max.npy'))
        attack=np.load(os.path.join(checkpointdir,'attack.npy'))
        self.gt_model=model
        self.min=torch.Tensor(min)
        self.max=torch.Tensor(max)
        self.attack=torch.Tensor(attack)
    def load_normal(self,checkpointdir):
        model = torch.load(os.path.join(checkpointdir, 'model.ckpt'))
        self.normal_model=model
    def load_improve(self,checkpointdir):
        model = torch.load(os.path.join(checkpointdir, 'model.ckpt'))
        self.improve_model=model

    def load_RPB(self,checkpointdir):
        model = torch.load(os.path.join(checkpointdir, 'model.ckpt'))
        self.RPB_model=model
    def prepare_dataset_loader(self,root='../data',transform=transforms.Compose([transforms.ToTensor()]),
                               train=True,batch_size=128,shuffle=True,num_workers=4):
        '''
        Input:

        Output:None
        '''
        name=self.dataset_name
        if train:
            self.trainset=get_dataset(name,root,transform,train)
            self.trainloader=torch.utils.data.DataLoader(self.trainset,batch_size=batch_size,shuffle=shuffle,
                                                         num_workers=num_workers)
        else:
            self.testset=get_dataset(name,root,transform,train)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=shuffle,
                                                           num_workers=num_workers)

    def normal_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,suffix='',img=None,target=None,method=None):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()
        logdir=os.path.join(logdir,self.dataset_name,'normal'+str(self.aug)+suffix)
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)

        normal_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda,
                     img,target,method,self.explain,explain_dir=logdir)
        writer.close()
        if self.normal_model is None:
            self.normal_model=model

        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'normal'+str(self.aug)+suffix)

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model
    def L1_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,alpha=0.01):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()
        logdir=os.path.join(logdir,self.dataset_name,'L1_'+str(alpha)+str(self.aug))
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)

        L1_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda,alpha)
        writer.close()

        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'L1_'+str(alpha)+str(self.aug))

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model
    def mixup_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,alpha=1):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()
        logdir=os.path.join(logdir,self.dataset_name, 'mixup_'+str(alpha)+str(self.aug))
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)

        mixup_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, alpha,self.use_cuda)
        writer.close()
        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'mixup_'+str(alpha)+str(self.aug))

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model
    def attack_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,inject_num=1,random=False,alpha=0.0,suffix='',
                     img=None,target=None,method=None):
        '''
        Input:

        Output: model
        '''
        self.inject_num=inject_num
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()

        if self.min is None:
            self.obtain_statistics()
        if self.attack is None:
            if not random:
                self.obtain_attack(inject_num=inject_num,alpha=alpha)
            else:
                self.random_attack(inject_num=inject_num)
        if random:
            r='_random'
        else:
            r=''
        logdir=os.path.join(logdir,self.dataset_name,'attack_'+str(inject_num)+'_'+str(alpha)+r+str(self.aug)+suffix)
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)

        attack_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda,
                     self.attack,self.min,self.max,img,target,method,self.explain,explain_dir=logdir)
        writer.close()
        if self.gt_model is None:
            self.gt_model=model

        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'attack_'+str(inject_num)+'_'+str(alpha)+r+str(self.aug)+suffix)

        self.save_gt(checkpointdir)
        return model

    def attack_img(self,img,label):
        attack_temp = self.attack[label]
        img = clip(img + attack_temp, self.min, self.max)
        return img

    def obtain_statistics(self):
        K=self.nclass[self.dataset_name]
        mu=None
        X2=None
        num = None  # numbers of samples except class
        mu_in=None
        X2_in=None
        num_in=None
        print('obtain statistics')
        for data, label in self.trainloader:
            C,H,W=data.size(1),data.size(2),data.size(3)
            self.channels,self.heights,self.width=C,H,W
            data_temp = data.permute([1, 0, 2, 3])
            data_temp = data_temp.reshape(C, -1)
            if self.min is None:
                self.min=torch.min(data_temp,1)[0].view(1,-1,1,1)
                self.max=torch.max(data_temp,1)[0].view(1,-1,1,1)
            else:
                m1=torch.cat([data_temp,self.min.view(-1,1)],1)
                m2=torch.cat([data_temp,self.max.view(-1,1)],1)
                self.min = torch.min(m1, 1)[0].view(1, -1, 1, 1)
                self.max = torch.max(m2, 1)[0].view(1, -1, 1, 1)

            if mu is None:
                mu=torch.zeros(K,C,H,W)
                X2=torch.zeros(K,C,H,W)
                num=torch.zeros(K,1,1,1)
                mu_in=torch.zeros(K,C,H,W)
                X2_in = torch.zeros(K, C, H, W)
                num_in = torch.zeros(K, 1, 1, 1)

            for i in range(K):
                temp=data[label!=i]
                mu[i]=mu[i]+torch.sum(temp,0,keepdim=True)
                X2[i]=X2[i]+torch.sum(temp**2,0,keepdim=True)
                num[i]+=temp.size(0)

                temp_in = data[label == i]
                if temp_in.size(0)>0:
                    mu_in[i] = mu[i] + torch.sum(temp_in, 0, keepdim=True)
                    X2_in[i] = X2[i] + torch.sum(temp_in ** 2, 0, keepdim=True)
                    num_in[i] += temp_in.size(0)

        self.mu=mu/num
        X2=X2/num
        self.var=X2-self.mu**2

        self.mu_in = mu_in / num_in
        X2_in = X2_in / num_in
        self.var_in = X2_in - self.mu_in ** 2

        print('min:',self.min,'max:',self.max)
        print('mean:',self.mu)
        print('var:',self.var)

        self.right_prob = torch.zeros(K, C, H, W, 2)
        epsilon=1e-4
        for data, label in self.trainloader:
            for i in range(K):
                temp_in = data[label == i]
                if temp_in.size(0) > 0:
                    temp_min=torch.sign(F.relu(self.min-temp_in+epsilon))
                    temp_max=torch.sign(F.relu(temp_in+epsilon-self.max))
                    self.right_prob[i,:,:,:,0]+=torch.sum(temp_min,0)
                    self.right_prob[i, :, :, :, 1] += torch.sum(temp_max, 0)
        self.right_prob=self.right_prob/num_in.view(K,1,1,1,1)
    def parallel(self):
        if self.normal_model is not None:
            self.normal_model=nn.DataParallel(self.normal_model)
        if self.gt_model is not None:
            self.gt_model=nn.DataParallel(self.gt_model)
        if self.improve_model is not None:
            self.improve_model=nn.DataParallel(self.improve_model)
        if self.RPB_model is not None:
            self.RPB_model=nn.DataParallel(self.RPB_model)
    def random_attack(self,inject_num=1):
        from scipy.stats import norm
        K,C,H,W=self.nclass[self.dataset_name],self.channels,self.heights,self.width
        self.attack = torch.zeros(K, C, H, W)
        jilu = torch.zeros(K)
        pan = torch.zeros(1, C, H, W, 2)
        print('find attack position ...')
        i=now=0
        maxnorm = (self.max - self.min).squeeze()
        while i<inject_num*K:
            index=int(np.random.rand()*K*C*H*W*2)
            now+=1
            n4=index %2
            index=int(index/2)
            n3=index % W
            index=int(index/W)
            n2=index %H
            index=int(index/H)
            n1=index % C
            index=int(index/C)
            n0=index
            if jilu[n0]<inject_num and pan[0,n1,n2,n3,n4]==0:
                jilu[n0]+=1
                pan[0, n1, n2, n3, n4]=1
                if n4==0:
                    self.attack[n0,n1,n2,n3]=-maxnorm[n1]*2
                else:
                    self.attack[n0, n1, n2, n3] = maxnorm[n1] * 2
                i+=1
                print('class',n0,'now',now)
        print(self.attack)


    def obtain_attack(self,inject_num=1,alpha=0.5):
        from scipy.stats import norm
        K,C,H,W=self.nclass[self.dataset_name],self.channels,self.heights,self.width
        eloss=torch.zeros(K,C,H,W,2)
        # 0: min 1:max
        sigma=torch.sqrt(self.var)+1e-8
        mu=self.mu
        T = 1 / (np.sqrt(2 * np.pi))

        t0=-(mu-self.min)/sigma
        phi0=torch.Tensor(norm.cdf(t0.numpy()))
        eloss[:,:,:,:,0]=sigma*(T*torch.exp(-0.5*t0**2)+t0*phi0)

        t1 = -(mu - self.max) / sigma
        phi1 = torch.Tensor(norm.cdf(t1.numpy()))
        eloss[:, :, :, :, 1] = -sigma * (-T * torch.exp(-0.5 * t1 ** 2) + t1 * (1-phi1))

        maxnorm=(self.max-self.min).view(-1)

        self.attack=torch.zeros(K,C,H,W)
        jilu=torch.zeros(K)
        pan=torch.zeros(1,C,H,W,2)
        pan[:,:,:,:,0]=1
        #========================================
        right_prob=self.right_prob
        '''
        right_prob=torch.zeros(K,C,H,W,2)
        sigma_in=torch.sqrt(self.var_in)+1e-8
        mu_in=self.mu_in
        t_in0 = -(mu_in - self.min) / sigma_in
        phi_in0 = torch.Tensor(norm.cdf(t_in0.numpy()))
        right_prob[:,:,:,:,0]=phi_in0

        t_in1 = -(mu_in - self.max) / sigma_in
        phi_in1 = torch.Tensor(norm.cdf(t_in1.numpy()))
        right_prob[:, :, :, :, 1] = 1-phi_in1
        '''
        #====================
        value=eloss*alpha+(1-alpha)*right_prob
        value_temp=value.view(-1)

        i=now=0
        print('sort start')
        sorted, indices = torch.sort(value_temp, descending=False)
        print('find attack position ...')
        while i<inject_num*K:
            index=indices[now]
            now+=1
            n4 = index % 2
            index = int(index / 2)
            n3 = index % W
            index = int(index / W)
            n2 = index % H
            index = int(index / H)
            n1 = index % C
            index = int(index / C)
            n0 = index
            if jilu[n0]<inject_num and pan[0,n1,n2,n3,n4]==0:
                jilu[n0]+=1
                #pan[0,n1,n2,n3,n4]=1
                pan[0, :, n2, n3, n4] = 1
                if n4==0:
                    self.attack[n0,n1,n2,n3]=-maxnorm[n1]*2
                else:
                    self.attack[n0, n1, n2, n3] = maxnorm[n1] * 2
                i+=1
                print('class',n0,'now',now)
        print(self.attack)

    def removeSPP_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,gamma=0.5):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()

        if self.mu is None:
            self.obtain_statistics()
        logdir=os.path.join(logdir,self.dataset_name,'removeSPP'+'_'+str(gamma)+str(self.aug))
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)
        mu=self.mu
        sigma=torch.sqrt(self.var)
        removeSPP_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda,
                     mu,sigma,prob=gamma)
        writer.close()

        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'removeSPP'+'_'+str(gamma)+str(self.aug))

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model, os.path.join(checkpointdir, 'model.ckpt'))
        return model
    def remove_attack_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,gamma=0.5,alpha=0.0):
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()

        if self.mu is None:
            self.obtain_statistics()
        if self.attack is None:
            self.obtain_attack(inject_num=1,alpha=alpha)
        logdir=os.path.join(logdir,self.dataset_name,'remove_attack'+'_'+str(gamma)+str(self.aug))
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)
        mu=self.mu
        sigma=torch.sqrt(self.var)
        remove_attack_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda,
                     mu,sigma,prob=gamma,attack=self.attack,min=self.min,max=self.max)
        writer.close()

        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'remove_attack'+'_'+str(gamma)+'_'+str(alpha)+str(self.aug))

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model, os.path.join(checkpointdir, 'model.ckpt'))
        return model

    def RPB_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,prob=0.5,alpha=0.2,point_size=1,suffix='',
                  img=None,target=None,method=None):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()

        logdir=os.path.join(logdir,self.dataset_name,'RPB_'+str(prob)+'_'+str(alpha)+'_'+str(point_size)+str(self.aug)+suffix)
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)
        from model.resnet_RPB import RPB
        rpb=RPB(prob=alpha,point_size=point_size)
        RPB_train(model,rpb, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda,
                  prob,alpha,img,target,method,self.explain,explain_dir=logdir)
        writer.close()
        if self.RPB_model is None:
            self.RPB_model=model
        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'RPB_'+str(prob)+'_'+str(alpha)+'_'
                                     +str(point_size)+str(self.aug)+suffix)

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model
    def RPB_batch_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,point_size=1,suffix=''):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()

        logdir=os.path.join(logdir,self.dataset_name,'RPB_batch'+'_'+str(point_size)+str(self.aug)+suffix)
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)
        from model.resnet_RPB import RPB_batch
        rpb=RPB_batch(point_size=point_size)
        RPB_train(model,rpb, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda)
        writer.close()
        if self.RPB_model is None:
            self.RPB_model=model
        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'RPB_batch'+'_'
                                     +str(point_size)+str(self.aug)+suffix)

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model
    def L1_RPB_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,prob=0.1,point_size=1,alpha=0.1):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()

        logdir=os.path.join(logdir,self.dataset_name,'L1_RPB_'+str(prob)+'_'+str(point_size)+'_'+str(alpha)+str(self.aug))
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)
        from model.resnet_RPB import RPB
        rpb=RPB(prob=prob,point_size=point_size)
        L1_RPB_train(model,rpb, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda,prob,alpha)
        writer.close()
        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'L1_RPB_'+str(prob)+'_'
                                     +str(point_size)+'_'+str(alpha)+str(self.aug))

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model
    def adversarial_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,perturbation_type='l2',eps=0.3):
        '''
        Input:

        Output: model
        '''
        if trainloader is None:
            trainloader=self.trainloader
        if testloader is None:
            testloader=self.testloader
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
        if self.use_cuda:
            model=model.cuda()
        logdir=os.path.join(logdir,self.dataset_name,'adversarial_'+perturbation_type+'_'+str(eps)+str(self.aug))
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)
        if self.min is None:
            self.obtain_statistics()
        adversarial_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer,
                          self.use_cuda,self.min,self.max,perturbation_type,eps)
        writer.close()

        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'adversarial_'+perturbation_type+'_'+str(eps)+str(self.aug))

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model
    def explain(self,img,label,logdir=None,model=None,method='GradientSHAP',attack=True,random=False,improve=False,suffix=''):
        '''
        input:
        img: batch X channels X height X width [BCHW], torch Tensor

        output:
        attribution_map: batch X height X width,numpy
        '''
        if not attack:
            if model is None:
                if improve:
                    model=self.improve_model
                else:
                    model=self.normal_model
        else:
            if model is None:
                if improve:
                    model = self.improve_model
                else:
                    model = self.gt_model
            img=self.attack_img(img,label)

        def weights_init(m):
            classname = m.__class__.__name__

            # print(classname)
            if classname.find('Conv') != -1:
                nn.init.xavier_normal_(m.weight.data)
            elif classname.find('Linear') != -1:
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
        if random:
            import copy
            random_model=copy.deepcopy(model)
            random_model.apply(weights_init)

        if self.use_cuda:
            img=img.cuda()
            label=label.cuda()

        def obtain_explain(alg,random):
            obj = alg.Explainer(model)
            mask = obj.get_attribution_map(img, label)
            mask = torch.mean(mask, 1, keepdim=True)
            if mask.requires_grad:
                mask=mask.detach()
            mask = mask.cpu().numpy()
            mask_random=None
            if random:
                obj = alg.Explainer(random_model)
                mask_random = obj.get_attribution_map(img, label)
                mask_random = torch.mean(mask_random, 1, keepdim=True)
                if mask_random.requires_grad:
                    mask_random = mask_random.detach()
                mask_random = mask_random.cpu().numpy()
            return mask,mask_random
        model=model.eval()
        if method=='GradientSHAP':
            from attribution_methods import GradientSHAP
            mask,mask_random=obtain_explain(GradientSHAP,random)
        elif method=='DeepLIFTSHAP':
            from attribution_methods import DeepLIFTSHAP
            mask,mask_random=obtain_explain(DeepLIFTSHAP,random)
        elif method=='Guided_BackProb':
            from attribution_methods import Guided_BackProp
            mask,mask_random=obtain_explain(Guided_BackProp, random)
        elif method=='DeepLIFT':
            from attribution_methods import DeepLIFT
            mask,mask_random=obtain_explain(DeepLIFT, random)
        elif method=='IntegratedGradients':
            from attribution_methods import IntegratedGradients
            mask,mask_random=obtain_explain(IntegratedGradients, random)
        elif method=='InputXGradient':
            from attribution_methods import InputXGradient
            mask,mask_random=obtain_explain(InputXGradient, random)
        elif method == 'Occlusion':
            from attribution_methods import Occlusion
            mask, mask_random = obtain_explain(Occlusion, random)
        elif method == 'Saliency':
            from attribution_methods import Saliency
            mask, mask_random = obtain_explain(Saliency, random)
        elif method=='GradCAM':
            from attribution_methods import Grad_CAM
            mask, mask_random = obtain_explain(Grad_CAM, random)
        elif method=='SmoothGrad':
            from attribution_methods import SmoothGrad
            mask, mask_random = obtain_explain(SmoothGrad, random)
        elif method=='RectGrad':
            from attribution_methods import RectGrad
            mask, mask_random = obtain_explain(RectGrad, random)
        else:
            print('no this method')

        if logdir is not None:
            if not os.path.exists(os.path.join(logdir,method+suffix)):  # 如果路径不存在
                os.makedirs(os.path.join(logdir,method+suffix))
            if img.requires_grad:
                img=img.detach()
            img=img.cpu().numpy()

            if attack:
                if self.min is not None:
                    save_images(img, os.path.join(logdir, method+suffix, 'raw_attack.png'), self.min.numpy(), self.max.numpy())
                if improve:
                    save_images(mask, os.path.join(logdir, method + suffix, 'mask_attack_improve.png'))
                    f=open(os.path.join(logdir, method + suffix, 'mask_attack_improve.txt'),'w')
                else:
                    save_images(mask, os.path.join(logdir, method+suffix, 'mask_attack.png'))
                    f = open(os.path.join(logdir, method+suffix, 'mask_attack.txt'), 'w')
                if random:
                    save_images(mask_random, os.path.join(logdir, method+suffix, 'mask_random_attack.png'))
            else:
                if self.min is not None:
                    save_images(img, os.path.join(logdir, method+suffix, 'raw.png'), self.min.numpy(), self.max.numpy())
                if improve:
                    save_images(mask, os.path.join(logdir, method+suffix, 'mask_improve.png'))
                    f = open(os.path.join(logdir, method+suffix, 'mask_improve.txt'), 'w')
                else:
                    save_images(mask, os.path.join(logdir, method+suffix, 'mask.png'))
                    f = open(os.path.join(logdir, method + suffix, 'mask.txt'), 'w')
                if random:
                    save_images(mask_random, os.path.join(logdir, method+suffix, 'mask_random.png'))
            if self.attack is not None:
                gt=torch.sign(torch.mean(torch.abs(self.attack[label]),1,keepdim=True)).numpy()
                Q_value=np.sum(np.abs(mask*gt))/np.sum(np.abs(mask)+1e-8)
                print('quantitative evaluation:', Q_value)
                print('quantitative evaluation:',Q_value,file=f)
            f.close()
            #cam=mask*0.5+img*0.5
            #save_images(cam, os.path.join(logdir, method, 'cam.jpg'))
            #if img.shape[0]==1:
            #    from utils.visualization import show_cam
            #    show_cam(img,mask, os.path.join(logdir, method, 'cam.jpg'))




