from __future__ import print_function
import torch
import torch.nn as nn
import math
import stoch_stream as stoch


def binarize(tensor):
    return tensor.sign()


# NOTE: Currently just a normal relu
class StochasticReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class Stochify(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, length=8):
        # save parameters before binarizing
        ctx.save_for_backward(input, weight, bias)
        api = stoch.to_stoch(input, length)
        apw = stoch.to_stoch(weight.t(), length)
        output = stoch.mm(api, apw, bipolar=True, output_stoch=False)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class StochLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(StochLinear, self).__init__()
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
        if bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return Stochify.apply(input, self.weight, self.bias)


class StochReLU(nn.Module):
    def __init__(self):
        super(StochReLU, self).__init__()

    def forward(self, input):
        return StochasticReLU.apply(input)
