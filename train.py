import ADGT
import os
from model import resnet,resnet_small,resnet_RPB,resnet_small_nobias
from utils import obtain_transform
import torch
import torchvision
from utils.AdamW import AdamW

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
ROOT='/newsd4/zgh/data'
CKPTDIR='/newsd4/zgh/ADGT/CKPT'
gamma=0.1
BATCHSIZE=128
AUG=False
MODEL='linear'#'resnet'#
CKPTDIR=os.path.join(CKPTDIR,MODEL)
DATASET_NAME='C10'#'MNIST'#'RestrictedImageNet'#'Flower102'#'C100'#
use_cuda=True
PROB=0.3
PLUGIN=0
MAX_EPOCH=50
wd=0
method=['SmoothGrad','InputXGradient','Guided_BackProb','Saliency','DeepLIFT','RectGrad','IntegratedGradients']
#method=['SmoothGrad']
torch.set_num_threads(4)
seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
import random,numpy
random.seed(seed)
numpy.random.seed(seed)
img=target=None

adgt=ADGT.ADGT(use_cuda=use_cuda,name=DATASET_NAME,aug=AUG)
transform_train,transform_test=obtain_transform.obtain_transform(DATASET_NAME)
adgt.prepare_dataset_loader(root=ROOT,train=False,transform=transform_test,batch_size=BATCHSIZE,shuffle=False)
adgt.prepare_dataset_loader(root=ROOT, train=True, transform=transform_test, batch_size=BATCHSIZE, shuffle=True)
K=adgt.nclass[DATASET_NAME]
NUMBER=K
iii=0
for data,label in adgt.trainloader:
    if img is None:
        img=data
        target=label
    else:
        img=torch.cat((img,data),0)
        target=torch.cat((target,label),0)
    iii+=1
    if iii>=3:
        break
imgtemp=[]
targettemp=[]
for i in range(K):
    imgtemp.append(img[target==i][0:NUMBER])
    targettemp.append(target[target==i][0:NUMBER])
img=torch.cat(tuple(imgtemp),0)
target=torch.cat(tuple(targettemp),0)
if AUG:
    adgt.prepare_dataset_loader(root=ROOT,train=True,transform=transform_train,batch_size=BATCHSIZE,shuffle=True)

if MODEL=='resnet':
    if adgt.dataset_name == 'MNIST':
        net = resnet_small.resnet18(indim=1, num_class=10)
    elif adgt.dataset_name =='C10':
        net=resnet_small.resnet18(indim=3,num_class=10)
        #net = resnet_small_nobias.resnet18(indim=3, num_class=10)
    elif adgt.dataset_name =='C100':
        #net=resnet_small.resnet18(indim=3,num_class=100)
        net = resnet.resnet50(num_classes=100)
    elif adgt.dataset_name =='Flower102':
        net = resnet.resnet50(num_classes=102)
        #net=resnet_RPB.resnet50(num_classes=102,prob=PROB,plugin_layer=PLUGIN)
    elif adgt.dataset_name == 'RestrictedImageNet':
        net = resnet.resnet50(num_classes=9)
        net = net.cuda()
        net=torch.nn.DataParallel(net)
elif MODEL == 'linear':
    if adgt.dataset_name == 'MNIST':
        net = resnet.GLM(in_features=28*28, out_features=10)
    elif adgt.dataset_name == 'C10':
        net = resnet.GLM(in_features=32*32*3, out_features=10)
    elif adgt.dataset_name == 'C100':
        net = resnet.GLM(in_features=32*32*3, out_features=100)
    elif adgt.dataset_name == 'Flower102':
        net = resnet.GLM(in_features=3, out_features=102)


if use_cuda:
    net=net.cuda()


import torch.nn as nn
def weights_init(m):
    classname = m.__class__.__name__

    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)



#suffix='wd'+str(wd)
suffix=MODEL
#suffix='flooding'
optimizer=AdamW(net.parameters(), lr=1e-3, betas=(0.5, 0.9),weight_decay=wd)

print('start attack train')
net=adgt.attack_train(net,'result',CKPTDIR,inject_num=1,alpha=0.0,suffix=suffix,img=img,target=target,method=method)
net.apply(weights_init)
print('start normal train')
net=adgt.normal_train(net,'result',CKPTDIR,suffix=suffix,img=img,target=target,method=method)
net.apply(weights_init)

print('start RPB train')
adgt.RPB_train(net,'result',CKPTDIR,prob=1,alpha=0.2,point_size=1,suffix=suffix,img=img,target=target,method=method)
net.apply(weights_init)


from utils.train import LabelSmoothing

'''
net=resnet_RPB.resnet50(num_classes=102,prob=PROB,plugin_layer=-1)
adgt.RPB_train(net,'result',CKPTDIR,prob=PROB,plugin_layer=-1)

net=resnet_RPB.resnet50(num_classes=102,prob=PROB,plugin_layer=1)
adgt.RPB_train(net,'result',CKPTDIR,prob=PROB,plugin_layer=1)
net=resnet_RPB.resnet50(num_classes=102,prob=PROB,plugin_layer=2)
adgt.RPB_train(net,'result',CKPTDIR,prob=PROB,plugin_layer=2)
net=resnet_RPB.resnet50(num_classes=102,prob=PROB,plugin_layer=3)
adgt.RPB_train(net,'result',CKPTDIR,prob=PROB,plugin_layer=3)

net=adgt.remove_attack_train(net,'result',CKPTDIR,gamma=0.1,alpha=0.0)
net.apply(weights_init)
net=adgt.remove_attack_train(net,'result',CKPTDIR,gamma=0.1,alpha=1.0)
net.apply(weights_init)


net=adgt.removeSPP_train(net,'result',CKPTDIR,gamma=0.1)
net.apply(weights_init)

net=adgt.removeSPP_train(net,'result',CKPTDIR,gamma=0.2)
net.apply(weights_init)


print('start improve',0.2)
net=adgt.removeSPP_train(net,'result',CKPTDIR,gamma=0.2)
net.apply(weights_init)
adgt.attack=None

net.apply(weights_init)
adgt.improve_model=None
print('start improve',0.1)
net=adgt.removeSPP_train(net,'result',CKPTDIR,gamma=0.1)
net.apply(weights_init)
adgt.improve_model=None
print('start improve',0.3)
net=adgt.removeSPP_train(net,'result',CKPTDIR,gamma=0.3)
net.apply(weights_init)


print('start attack train',0.0)
net=adgt.attack_train(net,'result',CKPTDIR,inject_num=1,alpha=0.0)
net.apply(weights_init)
print('start attack train',0.2)
net=adgt.attack_train(net,'result',CKPTDIR,inject_num=1,alpha=0.2)
net.apply(weights_init)
adgt.attack=adgt.gt_model=None
print('start attack train',0.4)
net=adgt.attack_train(net,'result',CKPTDIR,inject_num=1,alpha=0.4)
net.apply(weights_init)
adgt.attack=adgt.gt_model=None
print('start attack train',0.6)
net=adgt.attack_train(net,'result',CKPTDIR,inject_num=1,alpha=0.6)
net.apply(weights_init)
adgt.attack=adgt.gt_model=None
print('start attack train',0.8)
net=adgt.attack_train(net,'result',CKPTDIR,inject_num=1,alpha=0.8)
net.apply(weights_init)
adgt.attack=adgt.gt_model=None

print('start attack train',1.0)
net=adgt.attack_train(net,'result',CKPTDIR,inject_num=1,alpha=1.0)
'''









