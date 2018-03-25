from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import stoch_util as sb
import warnings


def binarize(tensor, deterministic=False):
    if deterministic:
        return tensor.sign()
    else:
        return (0.5 - sb.binarize(tensor.tanh()).type(type(tensor))).sign()


class BinarizedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, training=False):
        # save parameters before binarizing
        ctx.save_for_backward(input, weight)
        weight = binarize(weight)
        output = input.mm(weight.t())

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # print("Doing backward pass on FUNCTION")
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight, None


bmm = BinarizedMM.apply


class BinarizedLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=False):
        super(BinarizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))

        # Have the bias as a parameter to make it identitical to other Linear modules. Raise warning if used though
        if bias:
            warnings.warn("Bias is deprecated. Running with no bias", DeprecationWarning)

        # Do Xavier Glorot initialization
        n = input_features + output_features
        self.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, input):
        return bmm(input, self.weight, self.training)


class BinarizedHardTanH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, training=False):
        ctx.training = training
        ctx.org_input = input
        return binarize(input, deterministic=True)

    @staticmethod
    def backward(ctx, grad_output):
        # print("Doing backward pass on ACTIVATION")
        # Straight through estimator
        grad_input = grad_output.clone()
        grad_input[ctx.org_input.abs() > 1] = 0
        return grad_input, None


bhtanh = BinarizedHardTanH.apply
