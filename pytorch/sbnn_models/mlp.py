from __future__ import print_function
import torch
import torch.nn as nn
from modules import BinarizedLinear, bhtanh


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, npasses, bias=False, dp_hidden=0.5, momentum=0.1, epsilon=1e-6):
        super(Net, self).__init__()
        self.npasses = npasses
        self.input_features = input_features

        self.dense1 = BinarizedLinear(input_features, hidden_units, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_units, epsilon, momentum)
        self.drophidden1 = nn.Dropout(dp_hidden)

        self.dense2 = BinarizedLinear(hidden_units, hidden_units, bias=bias)
        self.bn2 = nn.BatchNorm1d(hidden_units, epsilon, momentum)
        self.drophidden2 = nn.Dropout(dp_hidden)

        self.dense3 = BinarizedLinear(hidden_units, hidden_units, bias=bias)
        self.bn3 = nn.BatchNorm1d(hidden_units, epsilon, momentum)
        self.drophidden3 = nn.Dropout(dp_hidden)

        self.dense4 = BinarizedLinear(hidden_units, output_features, bias=bias)

    def fpass(self, x):
        # Input layer
        x = x.view(-1, self.input_features)

        # 1st hidden layer
        x = self.dense1(x)
        x = self.bn1(x)
        x = bhtanh(x)
        x = self.drophidden1(x)

        # 2nd hidden layer
        x = self.dense2(x)
        x = self.bn2(x)
        x = bhtanh(x)
        x = self.drophidden2(x)

        # 3nd hidden layer
        x = self.dense3(x)
        x = self.bn3(x)
        x = bhtanh(x)
        x = self.drophidden3(x)

        # Output Layer
        x = self.dense4(x)
        return x

    def forward(self, x):
        if self.npasses >= 2:
            partial_output = self.fpass(x)
            for i in xrange(self.npasses - 2):
                partial_output += self.fpass(x)

            # When adding the partial result, detach it
            # This ensure that functional graph from the previous passes does not get autograd
            # Only one backward pass for npasses of forward pass
            output = self.fpass(x) + partial_output.detach()
            return output
        else:
            return self.fpass(x)

    def backward(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.back_clamp()

    def back_clamp(self):
        for module in self.modules():
            if isinstance(module, BinarizedLinear):
                module.weight.data.clamp_(-1, 1)
