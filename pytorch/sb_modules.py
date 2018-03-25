from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import stochbin as sb


def binarize(tensor, deterministic=False):
    if deterministic:
        return tensor.sign()
    else:
        return (0.5 - sb.binarize(tensor.tanh()).type(type(tensor))).sign()


class BinarizeActivation(torch.autograd.Function):
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


class BinarizeMulAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, training=False):
        # save parameters before binarizing
        ctx.save_for_backward(input, weight, bias)
        weight = binarize(weight)
        output = input.mm(weight.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # print("Doing backward pass on FUNCTION")
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None


class BinarizeLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(BinarizeLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Do Xavier Glorot initialization
        n = input_features + output_features
        self.weight.data.normal_(0, math.sqrt(2.0 / n))

        if bias:
            self.bias.data.zero_()

    def forward(self, input):
        return BinarizeMulAdd.apply(input, self.weight, self.bias, self.training)


class BinaryHardTanhH(nn.Module):
    def __init__(self):
        super(BinaryHardTanhH, self).__init__()

    def forward(self, input):
        return BinarizeActivation.apply(input, self.training)
