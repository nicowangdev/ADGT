from utils.prepare_dataset import get_dataset
from utils.train import normal_train
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils.visualization import save_images
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
        '''
        Input:

        Output:None
        '''
        self.dataset_name=name
        if train:
            self.trainset=get_dataset(name,root,transform,train)
            self.trainloader=torch.utils.data.DataLoader(self.trainset,batch_size=batch_size,shuffle=shuffle,
                                                         num_workers=num_workers)
        else:
            self.testset=get_dataset(name,root,transform,train)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=shuffle,
                                                           num_workers=num_workers)

    def normal_train(self,model,logdir,checkpointdir,trainloader=None,testloader=None,optimizer=None,
                     schedule=None,criterion=nn.CrossEntropyLoss(),max_epoch=50,):
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
        logdir=os.path.join(logdir,self.dataset_name,'normal')
        writer = SummaryWriter(log_dir=logdir )
        print('save logs to :', logdir)

        normal_train(model, trainloader, testloader, optimizer,schedule, criterion, max_epoch, writer, self.use_cuda)
        writer.close()
        if self.normal_model is None:
            self.normal_model=model

        checkpointdir = os.path.join(checkpointdir, self.dataset_name, 'normal')

        if not os.path.exists(checkpointdir):  # 如果路径不存在
            os.makedirs(checkpointdir)
        print('save checkpoints to :', checkpointdir)
        torch.save(model,os.path.join(checkpointdir,'model.ckpt'))
        return model

    def attack(self,img):
        return img

    def explain(self,img,logdir=None,model=None,method='SHAP',attack=False):
        '''
        input:
        img: batch X channels X height X width [BCHW], torch Tensor

        output:
        attribution_map: batch X height X width,numpy
        '''
        if not attack:
            if model is None:
                model=self.normal_model
        else:
            if model is None:
                model=self.gt_model
                img=self.attack(img)

        if self.use_cuda:
            img=img.cuda()

        if method=='SHAP':
            from attribution_methods import explainer
            obj=explainer.Explainer(model)
        mask =obj.get_attribution_map(img)

        if logdir is not None:
            img=img.cpu().numpy()
            save_images(img,os.path.join(logdir,method,'raw.jpg'))
            save_images(mask, os.path.join(logdir, method, 'mask.jpg'))
            #cam=mask*0.5+img*0.5
            #save_images(cam, os.path.join(logdir, method, 'cam.jpg'))
            if img.shape[0]==1:
                from utils.visualization import show_cam
                show_cam(img,mask, os.path.join(logdir, method, 'cam.jpg'))




