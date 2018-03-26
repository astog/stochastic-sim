from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.bn4 = nn.BatchNorm1d(10)

    def forward(self, x):
        # Conv layer 1
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(self.bn1(x))

        # Conv layer 2
        x = F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)
        x = F.relu(self.bn2(x))

        # FC layer 1
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(self.bn3(x))
        x = F.dropout(x, training=self.training)

        # FC layer 2
        x = self.bn4(self.fc2(x))
        return x

    def backward(self, loss, optimizer):
        # Zero grads, before calculation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
