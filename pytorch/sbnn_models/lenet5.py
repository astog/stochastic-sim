from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import BinarizedHardTanH, BinarizedConv2d, BinarizedLinear


class Net(nn.Module):
    def __init__(self, npasses, momentum=0.1, epsilon=1e-6):
        super(Net, self).__init__()
        self.npasses = npasses

        self.conv1 = BinarizedConv2d(1, 6, 5, deterministic=False, bias=True)
        self.bn1 = nn.BatchNorm2d(6, momentum=momentum, eps=epsilon)

        self.conv2 = BinarizedConv2d(6, 16, 5, deterministic=False, bias=True)
        self.bn2 = nn.BatchNorm2d(16, momentum=momentum, eps=epsilon)

        self.fc1 = BinarizedLinear(4*4*16, 120, deterministic=False, bias=True)
        self.bn3 = nn.BatchNorm1d(120, momentum=momentum, eps=epsilon)

        self.fc2 = BinarizedLinear(120, 84, deterministic=False, bias=True)
        self.bn4 = nn.BatchNorm1d(84, momentum=momentum, eps=epsilon)

        self.fc3 = BinarizedLinear(84, 10, deterministic=False, bias=True)
        self.bn5 = nn.BatchNorm1d(10, momentum=momentum, eps=epsilon)

        self.htanh = BinarizedHardTanH(deterministic=True)

    def fpass(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        x = self.htanh(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        x = self.htanh(x)

        x = x.view(-1, 4*4*16)
        x = self.bn3(self.fc1(x))
        x = self.htanh(x)

        x = self.bn4(self.fc2(x))
        x = self.htanh(x)

        x = self.bn5(self.fc3(x))
        return x

    def forward(self, x):
        # Save weights before binarization
        self.save_weights()
        accumulated_sum = self.fpass(x)
        for i in xrange(self.npasses - 1):
            accumulated_sum += self.fpass(x)

        # Note: Don't detach graph for n-1 passes, since pytorch will calculate
        # gradients for those passes and then accumulate them.
        # To maintain the Monte Carlo method of forward and backprop we need
        # gradients from each forward pass.
        output = accumulated_sum / float(self.npasses)

        # Restore real weights before leaving
        self.restore_weights()

        return output

    def backward(self, loss, optimizer):
        # Zero grads, before calculation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Clamp weights between [-1, 1]
        self.back_clamp()

    def save_weights(self):
        for module in self.modules():
            if hasattr(module, 'save_param'):
                module.save_param()

    def restore_weights(self):
        for module in self.modules():
            if hasattr(module, 'restore_param'):
                module.restore_param()

    def back_clamp(self):
        for module in self.modules():
            if isinstance(module, BinarizedLinear) or isinstance(module, BinarizedConv2d):
                module.weight.data.clamp_(-1, 1)
