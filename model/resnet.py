import torch
import torch.nn.functional as F
import torch.nn as nn
bn=nn.BatchNorm2d
class BatchNorm2d_fake(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.bn=nn.BatchNorm2d(out_channels,affine=False)
        self.alpha=nn.Parameter(torch.ones(1,out_channels,1,1))
        self.beta0=nn.Parameter(torch.zeros(1,out_channels,1,1))
        self.beta1=nn.Parameter(torch.zeros(1,out_channels,1,1))
    def forward(self, x):
        out=self.bn(x)*self.alpha
        return out
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34



    """

    # BasicBlock and BottleNeck block

    # have different output size

    # we use class attribute expansion

    # to distinct

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function

        self.residual_function = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),

            bn(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),

            bn(out_channels * BasicBlock.expansion)

        )

        # shortcut

        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function

        # use 1*1 convolution to match the dimension

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            if stride !=1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride,stride=stride),
                    nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, bias=False),
                    bn(out_channels * BasicBlock.expansion)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, bias=False),
                    bn(out_channels * BasicBlock.expansion)
                )

    def forward(self, x):
        #return self.residual_function(x) + self.shortcut(x)
        return F.relu(self.residual_function(x) + self.shortcut(x))
        #return self.residual_function(x)


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers



    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channels * BottleNeck.expansion),

        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BottleNeck.expansion * out_channels:
            if stride != 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels * BasicBlock.expansion)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels * BasicBlock.expansion)
                )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)

class DropAvg(nn.Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if self.training:
            p=torch.zeros(x.size(0),x.size(1)).cuda()+self.p
            prob=torch.bernoulli(p)
            return x*prob
        else:
            return x*self.p
class RandomAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if self.training:
            prob=torch.rand(x.size(0),x.size(1)).cuda()
            return x*prob
        else:
            return x*0.5
class ResNet(nn.Module):

    def __init__(self, block, num_block, indim=3,num_classes=10):
        super().__init__()

        self.in_channels = 64
        temp=self.in_channels

        self.conv1 = nn.Sequential(

            nn.Conv2d(indim, temp, kernel_size=3, padding=1, bias=False),

            bn(temp),

            nn.ReLU(inplace=True))

        # we use a different inputsize than the original paper

        # so conv2_x's stride is 1

        self.conv2_x = self._make_layer(block, temp, num_block[0], 1)

        self.conv3_x = self._make_layer(block, temp*2, num_block[1], 2)

        self.conv4_x = self._make_layer(block,temp*4, num_block[2], 2)

        #self.conv5_x = self._make_layer(block, temp*8, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu=nn.ReLU(inplace=True)
        lastdim=7
        if indim==1:
            lastdim=8
        self.fc = nn.Linear(temp*4 * block.expansion, num_classes,bias=True)
        self.weights_init()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))

            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Conv' or 'SNConv') != -1:
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        iii = torch.LongTensor(range(x.size(0))).cuda()
        x.requires_grad_(True)
        out1 = self.conv1(x)

        out2 = self.conv2_x(out1)
        out3 = self.conv3_x(out2)

        out4 = self.conv4_x(out3)

        #output = self.conv5_x(output)
        #output=self.relu(out4)

        output=out4
        output = self.avg_pool(output)

        h = output.view(output.size(0), -1)
        #h=self.drop(h)
        output = self.fc(h)
        return output


class Final(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d((1, 1))
        self.fc=nn.Linear(in_channels, num_classes,bias=False)
        self.relu=nn.ReLU(inplace=True)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Conv' or 'SNConv') != -1:
            m.weight.data.normal_(0.0, 0.02)
    def forward(self, x):
        x=self.gap(x)
        out=self.fc(x.view(x.size(0),-1))
        return out
def resnet18(indim=3,num_class=10):
    """ return a ResNet 18 object

    """

    return ResNet(BasicBlock, [2, 2, 2, 2],indim=indim,num_classes=num_class)


def resnet34():
    """ return a ResNet 34 object

    """

    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    """ return a ResNet 50 object

    """

    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    """ return a ResNet 101 object

    """

    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    """ return a ResNet 152 object

    """

    return ResNet(BottleNeck, [3, 8, 36, 3])