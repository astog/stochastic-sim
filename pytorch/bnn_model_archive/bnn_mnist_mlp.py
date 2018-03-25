from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from bnn_modules import BinarizeLinear, BinaryHardTanhH
import time
import datetime

# Training settings
parser = argparse.ArgumentParser(description='Stochastic BNN for MNIST MLP')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--test-batch-size', type=int, default=100)
parser.add_argument('--hunits', type=int, default=1024)
parser.add_argument('--momentum', type=float, default=0.1)
parser.add_argument('--dp-in', type=float, default=0.2)
parser.add_argument('--dp-hidden', type=float, default=0.5)
parser.add_argument('--epsilon', type=float, default=1e-4)
parser.add_argument('--dpath', type=str, default="./pytorch_data/")
parser.add_argument('--download', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=50)

args = parser.parse_args()

for arg in vars(args):
    print("{0:{1}<20} {2}".format(str(arg) + ' ', '-', getattr(args, arg)))
print("\n")

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Dataset transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# Cuda dataset arguments
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = datasets.MNIST(args.dpath, train=True, download=args.download, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_dataset = datasets.MNIST(args.dpath, train=False, download=args.download, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super(Net, self).__init__()
        self.hidden_units = hidden_units

        self.dropin = nn.Dropout(args.dp_in)
        self.dense1 = BinarizeLinear(input_features, self.hidden_units)
        self.bn1 = nn.BatchNorm1d(self.hidden_units, args.epsilon, args.momentum)
        self.actv1 = BinaryHardTanhH()
        self.drophidden1 = nn.Dropout(args.dp_hidden)

        self.dense2 = BinarizeLinear(self.hidden_units, self.hidden_units)
        self.bn2 = nn.BatchNorm1d(self.hidden_units, args.epsilon, args.momentum)
        self.actv2 = BinaryHardTanhH()
        self.drophidden2 = nn.Dropout(args.dp_hidden)

        self.dense3 = BinarizeLinear(self.hidden_units, self.hidden_units)
        self.bn3 = nn.BatchNorm1d(self.hidden_units, args.epsilon, args.momentum)
        self.actv3 = BinaryHardTanhH()
        self.drophidden3 = nn.Dropout(args.dp_hidden)

        self.dense4 = BinarizeLinear(self.hidden_units, output_features)
        self.bn4 = nn.BatchNorm1d(output_features, args.epsilon, args.momentum)

    def forward(self, x):
        # Input layer
        x = x.view(-1, 28 * 28)
        x = self.dropin(x)

        # 1st hidden layer
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.actv1(x)
        x = self.drophidden1(x)

        # 2nd hidden layer
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.actv2(x)
        x = self.drophidden2(x)

        # 3nd hidden layer
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.actv3(x)
        x = self.drophidden3(x)

        # Output Layer
        x = self.dense4(x)
        x = self.bn4(x)

        return x

    def back_clamp(self):
        self.dense1.weight.data.clamp_(-1, 1)
        if self.dense1.bias is not None:
            self.dense1.bias.data.clamp_(-1, 1)

        self.dense2.weight.data.clamp_(-1, 1)
        if self.dense2.bias is not None:
            self.dense2.bias.data.clamp_(-1, 1)

        self.dense3.weight.data.clamp_(-1, 1)
        if self.dense3.bias is not None:
            self.dense3.bias.data.clamp_(-1, 1)

        self.dense4.weight.data.clamp_(-1, 1)
        if self.dense4.bias is not None:
            self.dense4.bias.data.clamp_(-1, 1)


model = Net(28 * 28, 10, args.hunits)
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    # Initialize batchnorm and dropout layers for training
    model.train()

    # Logging variables
    train_batch_count = 0
    train_batch_avg_loss = 0
    train_batch_avg_count = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
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
        # model.back_clamp()

        train_batch_count += 1
        train_batch_avg_loss += float(loss)
        train_batch_avg_count += 1

        if batch_idx % args.log_interval == 0:
            print("Epoch: {: <6}\tBatches: {: 7.2f}%\t Average Batch Loss: {:.5e}".format(
                epoch, 100. * train_batch_count / len(train_loader),
                float(loss) / train_batch_avg_count
            ))
            train_batch_avg_loss = 0
            train_batch_avg_count = 0

    if train_batch_avg_loss > 0:
        print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.5e}".format(
            epoch, 100. * train_batch_count / len(train_loader),
            float(loss) / train_batch_avg_count
        ))


def test(epoch):
    # Initialize batchnorm and dropout layers for testing
    model.eval()

    # Logging variables
    correct = 0
    test_batch_count = 0
    test_batch_avg_loss = 0
    test_batch_avg_count = 0

    for batch_idx, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        loss = criterion(output, target).data[0]  # sum up batch loss

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_batch_count += 1
        test_batch_avg_loss += float(loss)
        test_batch_avg_count += 1

        if batch_idx % args.log_interval == 0:
            print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.5e}".format(
                epoch, 100. * test_batch_count / len(test_loader),
                float(loss) / test_batch_avg_count
            ))
            train_batch_avg_loss = 0
            test_batch_avg_count = 0

    if train_batch_avg_loss > 0:
        print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.5e}".format(
            epoch, 100. * test_batch_count / len(test_loader),
            float(loss) / test_batch_avg_count
        ))

    print('\nTest set accuracy: {}/{} ({:.4f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

    return correct


if __name__ == '__main__':
    correct = None
    for epoch in range(1, args.epochs + 1):
        time_start = time.clock()
        train(epoch)
        print("\n{:-<72}\n".format(""))
        correct = test(epoch)
        time_complete = time.clock() - time_start
        print("\nTime to complete epoch {} == {} sec(s)".format(
            epoch, time_complete
        ))
        print("Estimated time left == {}".format(
            str(datetime.timedelta(seconds=time_complete * (args.epochs - epoch)))
        ))

        print("{:=<72}\n".format(""))

    print("\nFinal test accuracy: {:.4f}".format(100. * correct / len(test_loader.dataset), end='\n\n'))
