from __future__ import print_function
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, channels_in, num_classes=10, bias=True, dp_dense=0.5):
        super(Net, self).__init__()
        self.include_bias = bias
        self.features = nn.Sequential(
            nn.Conv2d(channels_in, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Need to remove this since cifar-10 has a smaller image compared to imagenet
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features_out = 256 * 1 * 1
        self.classifier = nn.Sequential(
            nn.Dropout(p=dp_dense),
            nn.Linear(self.features_out, 256, bias=self.include_bias),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp_dense),
            nn.Linear(256, 256, bias=self.include_bias),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes, bias=self.include_bias),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.features_out)
        x = self.classifier(x)
        return x

    def backward(self, loss, optimizer):
        # Zero grads, before calculation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
