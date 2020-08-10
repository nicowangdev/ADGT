import ADGT
import os
from model import resnet
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

use_cuda=True
ROOT='/newsd4/zgh/data'
CKPTDIR='/newsd4/zgh/ADGT/CKPT'
adgt=ADGT.ADGT(use_cuda=use_cuda)
adgt.prepare_dataset_loader(root=ROOT,train=True,shuffle=True)
adgt.prepare_dataset_loader(root=ROOT,train=False,shuffle=False)
net=resnet.resnet18(indim=1)
if use_cuda:
    net=net.cuda()
net=adgt.normal_train(net,'result',CKPTDIR)
