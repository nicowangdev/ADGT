import ADGT
import os
from model import resnet
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

use_cuda=True
adgt=ADGT.ADGT(use_cuda=use_cuda)
adgt.prepare_dataset_loader(train=True,shuffle=True)
adgt.prepare_dataset_loader(train=False,shuffle=False)
net=resnet.resnet18(indim=1)
if use_cuda:
    net=net.cuda()
net=adgt.normal_train(net,'result')
