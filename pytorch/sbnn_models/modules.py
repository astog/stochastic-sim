from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import stoch_util as sb
import warnings


def binarize_(tensor, deterministic=False):
    if deterministic:
        tensor.sign_()
    else:
        mask = sb.binarize(tensor)
        tensor[mask] = -1
        tensor[1-mask] = 1


class BinarizedLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=False):
        # Have the bias as a parameter to make it identitical to other Linear modules. Raise warning if used though
        if bias:
            warnings.warn("Bias is deprecated. Running with no bias", DeprecationWarning)
            bias = False

        super(BinarizedLinear, self).__init__(input_features, output_features, bias)
        self.real_weight = self.weight.data.clone()

    def forward(self, input):
        self.weight.data.copy_(self.real_weight)
        binarize_(self.weight.data)

        return F.linear(input, self.weight)

    def save_param(self):
        # print(self.real_weight, self.weight.data)
        self.real_weight.copy_(self.weight.data)

    def restore_param(self):
        # print(self.weight.data, self.real_weight)
        self.weight.data.copy_(self.real_weight)


class BinarizedHardTanH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.org_input = input.clone()
        binarize_(input, deterministic=True)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # print("Doing backward pass on ACTIVATION")
        # Straight through estimator
        grad_input = grad_output.clone()
        grad_input[ctx.org_input.abs() > 1] = 0
        return grad_input, None


bhtanh = BinarizedHardTanH.apply
