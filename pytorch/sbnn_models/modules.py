from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import stoch_util as sb
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out


def binarize_(tensor, deterministic=False):
    if deterministic:
        tensor.sign_()
    else:
        mask = sb.binarize(tensor)
        tensor[mask] = -1
        tensor[1 - mask] = 1


def he_init(tensor, dist='uniform'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    n = fan_in

    if dist == "uniform":
        #
        tensor.uniform_(-1.0, 1.0)
        # Scale so that the final variace of this layer is 3/input_features
        tensor.mul_(np.sqrt(3.0 / n) / tensor.std())
    else:
        tensor.normal_(0.0, np.sqrt(3.0 / n))
    # print("Init variance : {:.7f} == {:.7f}".format(tensor.var(), 3.0 / n), end='\n\n')


class BinarizedLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=False, deterministic=False):
        self.deterministic = deterministic

        super(BinarizedLinear, self).__init__(input_features, output_features, bias)

        # nn.init.xavier_uniform(self.weight.data, gain=nn.init.calculate_gain('tanh'))
        he_init(self.weight.data, dist='normal')

        self.real_weight = self.weight.data.clone()

    def forward(self, input):
        self.weight.data.copy_(self.real_weight)
        binarize_(self.weight.data, deterministic=self.deterministic)
        return F.linear(input, self.weight, self.bias)

    def save_param(self):
        # print(self.real_weight, self.weight.data)
        self.real_weight.copy_(self.weight.data)

    def restore_param(self):
        # print(self.weight.data, self.real_weight)
        self.weight.data.copy_(self.real_weight)


class BinarizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, deterministic=False):
        super(BinarizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.deterministic = deterministic

        # nn.init.xavier_normal(self.weight.data, gain=nn.init.calculate_gain('tanh'))
        he_init(self.weight.data, dist='normal')

        self.real_weight = self.weight.data.clone()

    def forward(self, input):
        self.weight.data.copy_(self.real_weight)
        binarize_(self.weight.data, deterministic=self.deterministic)
        out = nn.functional.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def save_param(self):
        # print(self.real_weight, self.weight.data)
        self.real_weight.copy_(self.weight.data)

    def restore_param(self):
        # print(self.weight.data, self.real_weight)
        self.weight.data.copy_(self.real_weight)


class STHardTanH(torch.autograd.Function):
    def __init__(self):
        super(STHardTanH, self).__init__()

    @staticmethod
    def forward(ctx, input, deterministic):
        ctx.org_input = input
        ctx.deterministirc = deterministic
        output = input.clone()
        # output.tanh()
        binarize_(output, deterministic=deterministic)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Get data
        input = ctx.org_input
        deterministic = ctx.deterministirc
        grad_input = grad_output.clone()

        # Straight through estimator
        grad_input[input.abs() > 1] = 0
        # Scale by dtanh/dx
        # scaling_factor = 1.0 - input.tanh().pow(2.0)
        # grad_input.data *= scaling_factor

        return grad_input, None, None


class BinarizedHardTanH(nn.Module):
    def __init__(self, deterministic=True):
        super(BinarizedHardTanH, self).__init__()
        self.deterministic = deterministic

    def forward(self, input):
        return STHardTanH.apply(input, self.deterministic)


class StochasticTanH(nn.Module):
    def __init__(self):
        super(StochasticTanH, self).__init__()

    def forward(self, input):
        # TODO: Implement stochastic version for tanh, currently a mathematical tanh is very close approximation to it
        return F.tanh(input)


class ScaleLayer(nn.Module):
    def __init__(self, gamma=1e-3):
        super(ScaleLayer, self).__init__()
        self.gamma = nn.Parameter(torch.FloatTensor([gamma]))

    def forward(self, input):
        return input * self.gamma
