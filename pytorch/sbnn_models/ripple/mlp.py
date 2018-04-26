from __future__ import print_function
import torch
import torch.nn as nn
from ..modules import BinarizedLinear, BinarizedHardTanH

using_bnn = True
if using_bnn:
    print("Ripple Network modified to be BNN\n")


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, npasses, bias=False, dp_dense=0.1):
        super(Net, self).__init__()
        self.npasses = npasses
        self.input_features = input_features

        self.dense1 = BinarizedLinear(input_features, hidden_units, bias=bias, deterministic=using_bnn)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.drophidden1 = nn.Dropout(dp_dense)

        self.dense2 = BinarizedLinear(hidden_units, hidden_units, bias=bias, deterministic=using_bnn)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.drophidden2 = nn.Dropout(dp_dense)

        self.dense3 = BinarizedLinear(hidden_units, hidden_units, bias=bias, deterministic=using_bnn)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.drophidden3 = nn.Dropout(dp_dense)

        self.dense4 = BinarizedLinear(hidden_units, output_features, bias=bias, deterministic=using_bnn)
        self.bn4 = nn.BatchNorm1d(output_features)

        self.bhtanh = BinarizedHardTanH(deterministic=True)

    def fpass(self, x):
        # Input layer
        x = x.view(-1, self.input_features)

        # 1st hidden layer
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.bhtanh(x)
        x = self.drophidden1(x)

        # 2nd hidden layer
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.bhtanh(x)
        x = self.drophidden2(x)

        # 3nd hidden layer
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.bhtanh(x)
        x = self.drophidden3(x)

        # Output Layer
        x = self.dense4(x)
        x = self.bn4(x)
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

    def save_weights(self):
        for module in self.modules():
            if hasattr(module, 'save_param'):
                module.save_param()

    def restore_weights(self):
        for module in self.modules():
            if hasattr(module, 'restore_param'):
                module.restore_param()

    def get_regul_loss(self, mode='pow4'):
        total_loss = None
        for module in self.modules():
            if isinstance(module, BinarizedLinear):
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
