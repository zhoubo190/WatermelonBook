"""
Author: Victoria
Created on 2017.9.25 13:00
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self, cuda):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4096, 84)
        self.fc2 = nn.Linear(84, 10)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.use_cuda = cuda
        if cuda:
            self.cuda()

    def forward(self, x):
        #print x.size()
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.conv2(x)
        #print x.size()
        x = self.relu(x)
        x = x.view(-1, 4096)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

def train(model, train_loader, epochs):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if model.use_cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 1000 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if model.use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target).data[0]#Variable.data
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__=="__main__":
    train_loader = torch.utils.data.DataLoader(
                datasets.MNIST("./data", train=True,
                                              download=True, transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                  ])),
                batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                datasets.MNIST("./data", train=False,transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                  ])),
                batch_size=16, shuffle=True)
    cuda = True
    model = LeNet(cuda)
    train(model, train_loader, epochs=10)
    test(model, test_loader)