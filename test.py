import ADGT
import os
from model import resnet
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
use_cuda=True
method='SmoothGrad'#'GradCAM'#'GradientSHAP'#'InputXGradient'#'IntegratedGradients'#'DeepLIFT'#'DeepLIFTSHAP'#'Guided_BackProb'#'Saliency'
ROOT='/newsd4/zgh/data'
CKPTDIR='/newsd4/zgh/ADGT/CKPT'
gamma=0.4
torch.set_num_threads(4)
adgt=ADGT.ADGT(use_cuda=use_cuda,name='C10')
transform_train = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
                [ transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
adgt.prepare_dataset_loader(root=ROOT,train=True,batch_size=16,shuffle=True)
adgt.prepare_dataset_loader(root=ROOT,train=False,shuffle=False)
net=resnet.resnet18(indim=3,num_class=10)
if use_cuda:
    net=net.cuda()
#net=adgt.normal_train(net,'result',CKPTDIR)
#net=adgt.attack_train(net,'result',CKPTDIR,random=False,inject_num=2,gamma=0.0)
#net=adgt.removeSPP_train(net,'result',CKPTDIR,gamma=gamma)
#'''
checkpointdir = os.path.join(CKPTDIR, 'C10', 'attack_'+str(1)+'_'+str(0.4))
adgt.load_gt(checkpointdir)
checkpointdir = os.path.join(CKPTDIR,  'C10', 'normal')
adgt.load_normal(checkpointdir)
#'''
for data,label in adgt.trainloader:
    adgt.explain(data,label,'result',method=method,random=True)
    adgt.explain(data, label, 'result',method=method, random=True,attack=False)
    #adgt.explain(data, label, 'result', model=net,method=method, random=False, attack=False,suffix=str(gamma))
    break


