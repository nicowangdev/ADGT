import ADGT
import os
from model import resnet
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

use_cuda=True
ROOT='/newsd4/zgh/data'
CKPTDIR='/newsd4/zgh/ADGT/CKPT'
adgt=ADGT.ADGT(use_cuda=use_cuda,name='C10')
transform_train = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
                [ transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
adgt.prepare_dataset_loader(root=ROOT,train=True,shuffle=True)
adgt.prepare_dataset_loader(root=ROOT,train=False,shuffle=False)
net=resnet.resnet18(indim=3,num_class=10)
if use_cuda:
    net=net.cuda()
#net=adgt.normal_train(net,'result',CKPTDIR)
#net=adgt.attck_train(net,'result',CKPTDIR,random=False,inject_num=2,gamma=0.0)
checkpointdir = os.path.join(CKPTDIR, 'C10', 'attack_'+str(1)+'_'+str(0.4))
adgt.load_gt(checkpointdir)
for data,label in adgt.trainloader:
    adgt.explain(data,label,'result')
    break