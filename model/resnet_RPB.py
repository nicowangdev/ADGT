import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class RPB_old(nn.Module):
    def __init__(self, num_features, height, width=None, prob=0.1, momentum=0.999, point_size=4):
        super(RPB_old, self).__init__()
        if width is None:
            width = height
        self.ps=point_size
        w=int(width/point_size)
        h=int(height/point_size)
        w2 = int(width / 2)
        h2 = int(height / 2)
        self.register_buffer('running_mean_1', torch.zeros(1, num_features, height, width))
        self.register_buffer('running_var_1', torch.ones(1, num_features, height, width))
        self.register_buffer('running_mean_2', torch.zeros(1, num_features, h2, w2))
        self.register_buffer('running_var_2', torch.ones(1, num_features, h2, w2))
        self.register_buffer('running_mean', torch.zeros(1, num_features, h, w))
        self.register_buffer('running_var', torch.ones(1, num_features, h, w))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.reset_running_stats()
        self.momentum = momentum
        self.prob = prob*0.3

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def forward(self, x):
        if not self.training:
            return x

        input=F.avg_pool2d(x,kernel_size=self.ps,stride=self.ps)
        self.running_mean = self.running_mean * self.momentum + torch.mean(input, 0, keepdim=True).data * (1 - self.momentum)
        self.running_var = self.running_var * self.momentum + torch.mean((input - self.running_mean).data ** 2, 0,
                                                                         keepdim=True) * (1 - self.momentum)
        R = torch.randn_like(input) * torch.sqrt(self.running_var) + self.running_mean
        P = torch.bernoulli(torch.ones_like(input) * self.prob)
        R=F.interpolate(R,scale_factor=self.ps)
        P=F.interpolate(P,scale_factor=self.ps)
        output = x * (1 - P) + R * P
        #================================
        input2 = F.avg_pool2d(x, kernel_size=2, stride=2)
        self.running_mean_2 = self.running_mean_2 * self.momentum + torch.mean(input2, 0, keepdim=True).data * (
                    1 - self.momentum)
        self.running_var_2 = self.running_var_2 * self.momentum + torch.mean((input2 - self.running_mean_2).data ** 2, 0,
                                                                         keepdim=True) * (1 - self.momentum)
        R2 = torch.randn_like(input2) * torch.sqrt(self.running_var_2 ) + self.running_mean_2
        P2 = torch.bernoulli(torch.ones_like(input2) * self.prob)
        R2 = F.interpolate(R2, scale_factor=2)
        P2 = F.interpolate(P2, scale_factor=2)
        output = output * (1 - P2) + R2 * P2
        #====================================
        self.running_mean_1 = self.running_mean_1 * self.momentum + torch.mean(x, 0, keepdim=True).data * (
                1 - self.momentum)
        self.running_var_1 = self.running_var_1 * self.momentum + torch.mean((x - self.running_mean_1).data ** 2, 0,
                                                                             keepdim=True) * (1 - self.momentum)
        R1 = torch.randn_like(x) * torch.sqrt(self.running_var_1) + self.running_mean_1
        P1 = torch.bernoulli(torch.ones_like(x) * self.prob)
        output = output * (1 - P1) + R1 * P1
        return output

class RPB(nn.Module):
    def __init__(self, num_features=None, height=None, width=None, prob=0.1, momentum=0.999, point_size=4):
        super(RPB, self).__init__()
        self.point_size=point_size
        self.prob=prob
    def forward(self, x):
        if not self.training:
            return x
        ps=self.point_size
        output=x
        #index = (torch.rand(x.size(0)) * x.size(0)).long().cuda()
        #R = x[index]
        while ps>0:
            index = (torch.rand(x.size(0)) * x.size(0)).long().cuda()
            R = x[index]
            P = torch.bernoulli(torch.ones(x.size(0),x.size(1),int(x.size(2)/ps)+1,int(x.size(3)/ps)+1) * self.prob).cuda()
            temp = F.interpolate(P, scale_factor=ps)
            rh=torch.rand(x.size(0))*ps
            rw=torch.rand(x.size(0))*ps
            rh,rw=rh.long().cuda(),rw.long().cuda()
            #P=torch.zeros_like(x)
            #for i in range(x.size(0)):
            #    P[i,:,:,:]=temp[i,:,rh[i]:rh[i]+x.size(2),rw[i]:rw[i]+x.size(3)]
            P=temp[:,:,rh[0]:rh[0]+x.size(2),rw[0]:rw[0]+x.size(3)]
            output=output * (1 - P) + R * P
            ps=int(ps/2)
        return output
import numpy as np
class RPB_batch(nn.Module):
    def __init__(self, num_features=None, height=None, width=None, prob=0.1, momentum=0.999, point_size=10):
        super(RPB_batch, self).__init__()
        self.point_size=point_size
        self.prob=prob
    def forward(self, x):
        if not self.training:
            return x
        ps=self.point_size
        output=x
        #index = (torch.rand(x.size(0)) * x.size(0)).long().cuda()
        index=np.arange(output.size(0))
        np.random.shuffle(index)
        index=torch.LongTensor(index).cuda()

        R = x[index]
        P = torch.zeros_like(x)
        P[:,:,0:ps,0:ps]=1
        output=output * (1 - P) + R * P
        return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,size=224,prob=0.1,plugin_layer=1,point_size=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.plugin=plugin_layer
        self.RPB0 = RPB(3, size, prob=prob,point_size=point_size)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.RPB1 = RPB(64, int(size/4), prob=prob)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.RPB2 = RPB(64* block.expansion, int(size / 4), prob=prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.RPB3 = RPB(128* block.expansion, int(size / 8), prob=prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.RPB4 = RPB(256* block.expansion, int(size / 16), prob=prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.plugin==0:
            x=self.RPB0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #x = self.maxpool(x)
        x=self.maxpool(x)

        if self.plugin == 1:
            x = self.RPB1(x)
        x = self.layer1(x)
        if self.plugin==2:
            x=self.RPB2(x)
        x = self.layer2(x)
        if self.plugin==3:
            x=self.RPB3(x)
        x = self.layer3(x)
        if self.plugin==4:
            x=self.RPB4(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
