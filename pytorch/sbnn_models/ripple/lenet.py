from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import BinarizedHardTanH, BinarizedConv2d, BinarizedLinear, ScaleLayer

using_bnn = True
if using_bnn:
    print("Using BNN")


lenet_model = [6, 16, 120, 84]


class Net(nn.Module):
    def __init__(self, channels_in, output_classes, npasses, infl_ratio, bias=False):
        super(Net, self).__init__()
        self.npasses = npasses

        for i in xrange(len(lenet_model)):
            lenet_model[i] = int(round(lenet_model[i] * infl_ratio))

        print("Lenet config\n", lenet_model, end='\n\n')

        self.conv1 = nn.Conv2d(channels_in, lenet_model[0], 5, bias=bias)
        self.bn1 = nn.BatchNorm2d(lenet_model[0])

        self.conv2 = BinarizedConv2d(lenet_model[0], lenet_model[1], 5, deterministic=using_bnn, bias=bias)
        self.bn2 = nn.BatchNorm2d(lenet_model[1])

        self.features_in = ((7 + channels_in) * (7 + channels_in) * lenet_model[1]) / (2 ** 2)

        self.fc3 = BinarizedLinear(self.features_in, lenet_model[2], deterministic=using_bnn, bias=bias)
        self.bn3 = nn.BatchNorm1d(lenet_model[2])

        self.fc4 = BinarizedLinear(lenet_model[2], lenet_model[3], deterministic=using_bnn, bias=bias)
        self.bn4 = nn.BatchNorm1d(lenet_model[3])

        self.fc5 = BinarizedLinear(lenet_model[3], output_classes, deterministic=using_bnn, bias=bias)
        self.sl5 = ScaleLayer(gamma=1e-3)

        self.htanh = BinarizedHardTanH(deterministic=True)

    def fpass(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x, inplace=True)
        x = self.bn1(x)
        x = self.htanh(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.htanh(x)

        x = x.view(-1, self.features_in)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.htanh(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.htanh(x)

        x = self.fc5(x)
        x = self.sl5(x)
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

    def get_regul_loss(self, mode='pow4'):
        total_loss = None
        for module in self.modules():
            if isinstance(module, BinarizedLinear) or isinstance(module, BinarizedConv2d):
                # Do bipolar regularization from Tang et al.
                if mode == 'tang':
                    loss = (1.0 - torch.pow(module.weight, 2)).sum()
                # Do bipolar regularization similar to l1, but shifted and flipped
                elif mode == 'bp1':
                    loss = torch.abs(torch.abs(module.weight) - 1).sum()
                # Do bipolar regularization similar to bp1, but smoothed at +1/-1
                elif mode == 'bp2':
                    loss = torch.pow(torch.abs(module.weight) - 1, 2).sum()
                # Do bipolar regularization using 4-degree function
                elif mode == 'pow4':
                    loss = (torch.pow(module.weight - 1, 2) * torch.pow(module.weight + 1, 2)).sum()
                else:
                    raise(RuntimeError, "Invalid regulation mode given to model")

                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss

        if total_loss is None:
            total_loss = 0.0

        return total_loss
