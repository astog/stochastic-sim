from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules import StochasticTanH, BinarizedConv2d, BinarizedLinear, ScaleLayer


class Net(nn.Module):
    def __init__(self, channels_in, output_classes, npasses, bias=False):
        super(Net, self).__init__()
        self.npasses = npasses

        # self.conv1 = BinarizedConv2d(channels_in, 6, 5, deterministic=False, bias=bias)
        self.conv1 = nn.Conv2d(channels_in, 6, 5, bias=bias)
        self.bn1 = nn.BatchNorm2d(6)
        self.actv1 = nn.Hardtanh()

        self.conv2 = BinarizedConv2d(6, 16, 5, deterministic=False, bias=bias)
        self.bn2 = nn.BatchNorm2d(16)
        self.actv2 = nn.Hardtanh()

        self.features_in = ((7 + channels_in) * (7 + channels_in) * 16) / (2 ** 2)

        self.fc1 = BinarizedLinear(self.features_in, 120, deterministic=False, bias=bias)
        self.bn3 = nn.BatchNorm1d(120)
        self.actv3 = nn.Hardtanh()

        self.fc2 = BinarizedLinear(120, 84, deterministic=False, bias=bias)
        self.bn4 = nn.BatchNorm1d(84)
        self.actv4 = nn.Hardtanh()

        self.fc3 = BinarizedLinear(84, output_classes, deterministic=False, bias=bias)
        # self.bn5 = nn.BatchNorm1d(output_classes)
        self.sl1 = ScaleLayer(gamma=1e-3)

        # self.tanh = StochasticTanH()

    def do_npass(self, x, *layers):
        # Save weights before binarization
        for layer in layers:
            self.save_weights(layer)

        # NOTE: Don't clone here, since forward pass on model does not edit input inplace.
        x_temp = x
        for layer in layers:
            x_temp = layer(x_temp)
        accumulated_sum = x_temp

        for i in xrange(self.npasses - 1):
            x_temp = x
            for layer in layers:
                x_temp = layer(x_temp)

            accumulated_sum += x_temp

        # Note: Don't detach graph for n-1 passes, since pytorch will calculate
        # gradients for those passes and then accumulate them.
        # To maintain the Monte Carlo method of forward and backprop we need
        # gradients from each forward pass.
        output = accumulated_sum / float(self.npasses)

        # Restore real weights before leaving
        for layer in layers:
            self.restore_weights(layer)

        return output

    def forward(self, x):
        x = self.do_npass(x, self.conv1)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.actv1(x)

        x = self.do_npass(x, self.conv2)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.actv2(x)

        x = x.view(-1, self.features_in)
        x = self.do_npass(x, self.fc1)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.actv3(x)

        x = self.do_npass(x, self.fc2)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.actv4(x)

        x = self.do_npass(x, self.fc3)
        # x = self.bn5(x)
        x = self.sl1(x)
        return x

    def backward(self, loss, optimizer):
        # Zero grads, before calculation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Clamp weights between [-1, 1]
        self.back_clamp()

    def save_weights(self, layer):
        if hasattr(layer, 'save_param'):
            layer.save_param()

    def restore_weights(self, layer):
        if hasattr(layer, 'restore_param'):
            layer.restore_param()

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
