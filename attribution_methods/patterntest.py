import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from attribution_methods.PatternNet import *
import numpy as np


def imshow(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
test_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=25, shuffle=False)


def train(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return net
    # torch.save(net.state_dict(), "parameters/" + 'cnn1.pt')


if __name__ == "__main__":
    # train()
    # prepare model
    net = Net()
    # model.load_state_dict(torch.load("parameters/cnn1.pt"))
    model = train(net)
    explainer = Explainer(model, trainloader)
    for batch, _ in test_dataloader:
        plt.figure()
        # batch_org = torch.empty(1, batch.shape[1], batch.shape[2], batch.shape[3])
        # batch_org[0] = batch[0]
        # batch1 = batch[0]
        # # batch1 = torch.cat((batch1, batch1, batch1), 0)
        # batch1 = batch1.numpy()u

        # plt.imshow(np.transpose(batch1, (1, 2, 0)))
        imshow(make_grid(batch, nrow=5))

        plt.figure()
        signal = explainer.get_pattern(batch.to(device))
        #
        signal = signal / 2 + .5
        #
        imshow(make_grid(signal, nrow=5))

        plt.show()
        break
