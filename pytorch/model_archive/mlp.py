from __future__ import print_function
import torch
import torch.nn as nn

using_bnn = False


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, bias=False, dp_dense=0.5):
        super(Net, self).__init__()
        self.input_features = input_features

        self.dense1 = nn.Linear(input_features, hidden_units, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.drophidden1 = nn.Dropout(dp_dense)

        self.dense2 = nn.Linear(hidden_units, hidden_units, bias=bias)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.drophidden2 = nn.Dropout(dp_dense)

        self.dense3 = nn.Linear(hidden_units, hidden_units, bias=bias)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.drophidden3 = nn.Dropout(dp_dense)

        self.dense4 = nn.Linear(hidden_units, output_features, bias=bias)
        self.bn4 = nn.BatchNorm1d(output_features)

        self.bhtanh = nn.Tanh()

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
        return self.fpass(x)

    def backward(self, loss, optimizer):
        # Zero grads, before calculation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Clamp weights between [-1, 1]
        self.back_clamp()

    def back_clamp(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.clamp_(-1.0, 1.0)
                # weight --> copies into --> real_weight before forward pass
                # module.real_weight.clamp_(-1.0, 1.0)
