from __future__ import print_function
import torch
import torch.nn as nn
from ..modules import BinarizedLinear, StochasticTanH


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, npasses, bias=False, dp_dense=0.1):
        super(Net, self).__init__()
        self.npasses = npasses
        self.input_features = input_features

        self.dense1 = BinarizedLinear(input_features, hidden_units, bias=bias, deterministic=False)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.drophidden1 = nn.Dropout(dp_dense)

        self.dense2 = BinarizedLinear(hidden_units, hidden_units, bias=bias, deterministic=False)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.drophidden2 = nn.Dropout(dp_dense)

        self.dense3 = BinarizedLinear(hidden_units, hidden_units, bias=bias, deterministic=False)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.drophidden3 = nn.Dropout(dp_dense)

        self.dense4 = BinarizedLinear(hidden_units, output_features, bias=bias, deterministic=False)
        self.bn4 = nn.BatchNorm1d(output_features)

        self.tanh = StochasticTanH()

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
        # Input layer
        x = x.view(-1, self.input_features)

        # 1st hidden layer
        x = self.do_npass(x, self.dense1)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.drophidden1(x)

        # 2nd hidden layer
        x = self.do_npass(x, self.dense2)
        x = self.bn2(x)
        x = self.tanh(x)
        x = self.drophidden2(x)

        # 3nd hidden layer
        x = self.do_npass(x, self.dense3)
        x = self.bn3(x)
        x = self.tanh(x)
        x = self.drophidden3(x)

        # Output Layer
        x = self.do_npass(x, self.dense4)
        x = self.bn4(x)
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
            if isinstance(module, BinarizedLinear):
                module.weight.data.clamp_(-1.0, 1.0)
                # weight  --> copies into --> real_weight before forward pass
                # module.real_weight.clamp_(-1.0, 1.0)
