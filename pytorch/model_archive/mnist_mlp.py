from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, npasses, bias=False, dp_hidden=0.5, momentum=0.1, epsilon=1e-6):
        super(Net, self).__init__()
        self.npasses = npasses
        self.input_features = input_features
        self.training_epoch = 0

        self.dense1 = nn.Linear(input_features, hidden_units, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_units, epsilon, momentum)
        self.drophidden1 = nn.Dropout(dp_hidden)

        self.dense2 = nn.Linear(hidden_units, hidden_units, bias=bias)
        self.bn2 = nn.BatchNorm1d(hidden_units, epsilon, momentum)
        self.drophidden2 = nn.Dropout(dp_hidden)

        self.dense3 = nn.Linear(hidden_units, hidden_units, bias=bias)
        self.bn3 = nn.BatchNorm1d(hidden_units, epsilon, momentum)
        self.drophidden3 = nn.Dropout(dp_hidden)

        self.dense4 = nn.Linear(hidden_units, output_features, bias=bias)
        self.bn4 = nn.BatchNorm1d(output_features, epsilon, momentum)

    def fpass(self, x):
        # Input layer
        x = x.view(-1, self.input_features)

        # 1st hidden layer
        x = self.dense1(x)
        x = F.tanh(self.bn1(x))
        x = self.drophidden1(x)

        # 2nd hidden layer
        x = self.dense2(x)
        x = F.tanh(self.bn2(x))
        x = self.drophidden2(x)

        # 3nd hidden layer
        x = self.dense3(x)
        x = F.tanh(self.bn3(x))
        x = self.drophidden3(x)

        # Output Layer
        x = self.dense4(x)
        x = self.bn4(x)

        # Logits output. coz of pytorch CrossEntropy has a builin log-softmax
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
                module.weight.data.clamp(-1, 1)
