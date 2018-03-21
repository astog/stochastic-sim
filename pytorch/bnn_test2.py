from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from progressbar import printProgressBar
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules2 import BinarizeLinear, BinaryHardTanhH
import time

# from models.binarized_modules import  Binarize,Ternarize,Ternarize2,Ternarize3,Ternarize4,HingeLoss

# data = '/media/gaurav/Data/udata/pytorch_data/'
data = '/home/gaurav/pytorch_data/'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


# BN parameters
print("batch_size = " + str(args.batch_size))
# alpha is the exponential moving average factor
print("alpha = " + str(args.momentum))
epsilon = 1e-4
print("epsilon = " + str(epsilon))

# MLP parameters
num_units = 1024
print("num_units = " + str(num_units))

# Training parameters
print("num_epochs = " + str(args.epochs))

# Dropout parameters
dropout_in = .2  # 0. means no dropout
print("dropout_in = " + str(dropout_in))
dropout_hidden = .5
print("dropout_hidden = " + str(dropout_hidden))

# Decaying LR
LR_start = .03
print("LR_start = " + str(LR_start))
LR_fin = 0.0003
print("LR_fin = " + str(LR_fin))
LR_decay = (LR_fin / LR_start)**(1. / args.epochs)
print("LR_decay = " + str(LR_decay), end='\n\n')
# BTW, LR decay might good for the BN moving average...


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super(Net, self).__init__()
        self.hidden_units = hidden_units

        self.dense1 = BinarizeLinear(input_features, self.hidden_units)
        self.actv1 = BinaryHardTanhH()  # Outputs -1, 1. Does straight through estimator in backprop

        self.dense4 = BinarizeLinear(self.hidden_units, output_features)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input layer
        x = x.view(-1, 28 * 28)

        # 1st hidden layer
        x = self.dense1(x)
        x = self.actv1(x)

        # Output Layer
        x = self.dense4(x)
        return x

    def back_clamp(self):
        self.dense1.weight.data.clamp_(-1, 1)
        if self.dense1.bias is not None:
            self.dense1.bias.data.clamp_(-1, 1)

        self.dense4.weight.data.clamp_(-1, 1)
        if self.dense4.bias is not None:
            self.dense4.bias.data.clamp_(-1, 1)


model = Net(28 * 28, 10, num_units)
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR_start)


def forward_pass(model, data):
    partial_output = model(data)
    for i in xrange(6):
        partial_output += model(data)

    # When adding the partial result, detach it
    # This ensure that functional graph from the first 7 bits does not get autograd
    output = (model(data) + partial_output.detach()) / 8.0
    return output


def train(epoch):
    model.train()
    print("({}) LR: {:.2e}".format(epoch, optimizer.param_groups[0]['lr']))
    train_batch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()

        # 2: Do optimization step
        optimizer.step()

        # 3: Make sure to clip weights / biases between -1, 1.
        model.back_clamp()

        train_batch_loss = 0.9*train_batch_loss + 0.1*loss.data[0]
        printProgressBar((batch_idx + 1) * len(data), len(train_loader.dataset), length=50)
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))

    # Decay LR
    optimizer.param_groups[0]['lr'] *= LR_decay


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = forward_pass(model, data)
        test_loss += criterion(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct


if __name__ == '__main__':
    max_correct = 0
    for epoch in range(1, args.epochs + 1):
        time_start = time.clock()
        train(epoch)
        max_correct = max(test(), max_correct)
        time_end = time.clock() - time_start
        print("Took time {} sec(s)\n".format(time_end))

    print("\n===========================\n\nBEST TEST ACCURACY: {:.4f} - {}/{}\n".format(
        100. * max_correct / len(test_loader.dataset), max_correct, len(test_loader.dataset)))
