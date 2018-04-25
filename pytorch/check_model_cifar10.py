from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from Adam import Adam
from dataloader import KFoldDataset

# Models
# from model_archive.mlp import Net
# from model_archive.lenet import Net
# from sbnn_models.ripple.mlp import Net
# from sbnn_models.ripple.lenet import Net
# from sbnn_models.wave.mlp import Net
from sbnn_models.wave.lenet import Net

import time
import datetime
import math
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

means = (125.307 / 255., 122.95 / 255., 113.865 / 255.)
stds = (62.9932 / 255., 62.0887 / 255., 66.7048 / 255.)

# Training settings
parser = argparse.ArgumentParser(description='Main testing script for CIFAR10 dataset')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--no-shuffle', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--hunits', type=int, default=32)
parser.add_argument('--include-bias', action='store_true', default=False)
parser.add_argument('--npasses', type=int, default=1)
parser.add_argument('--cpasses', type=int, default=32)
parser.add_argument('--dp-dense', type=float, default=0.1)
parser.add_argument('--dp-conv', type=float, default=0.0)
parser.add_argument('--dpath', type=str, default="./pytorch_data/")
parser.add_argument('--download', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--check', type=str, default=None)

args = parser.parse_args()

for arg in vars(args):
    print("{0:{1}<20} {2}".format(str(arg) + ' ', '-', getattr(args, arg)))
print("\n")

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.shuffle = not args.no_shuffle

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Dataset transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

# Cuda dataset arguments
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = datasets.CIFAR10(args.dpath, train=True, download=args.download, transform=transform_train)
test_dataset = datasets.CIFAR10(args.dpath, train=False, download=args.download, transform=transform_test)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

# model = Net(28 * 28, 10, args.hunits, args.npasses, bias=False, dp_dense=args.dp_dense)
model = Net(3, 10, args.npasses, bias=args.include_bias)
# model = Net(3, 10, bias=args.include_bias)

if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()

criterion = nn.CrossEntropyLoss()


def test(cpasses):
    # Initialize batchnorm and dropout layers for testing
    model.eval()

    # Logging variables
    correct = 0

    print("{: <5}  --  Testing".format(cpasses))
    for batch_idx, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data, volatile=True)

        output = None
        for i in xrange(cpasses):
            if output is None:
                output = model(data)
            else:
                output += model(data)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()

        if batch_idx % args.log_interval == 0:
            print("| {:3.4f}%".format(100. * batch_idx / len(test_loader)), end='\r')

    print("| {:3.4f}%".format(100. * batch_idx / len(test_loader)))

    print('\nTest accuracy: {}/{} ({:.4f}%)'.format(
        correct, len(test_loader.dataset),
        (100. * correct) / len(test_loader.dataset)
    ))

    return correct


if __name__ == '__main__':
    if args.check is not None:
        print("Loading module", args.check)
        model.load_state_dict(torch.load(args.check))

        test_accuracy = np.zeros(args.cpasses)
        for cpasses in xrange(args.cpasses):
            test_accuracy[cpasses] = (float(test(cpasses+1)) / (len(test_loader) * args.batch_size))


        '''
        # best fit of data
        (mu, sigma) = norm.fit(test_accuracy)

        # the histogram of the data
        n, bins, patches = plt.hist(test_accuracy, 60, normed=1, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)

        # plot
        plt.xlabel('Test accuracy')
        plt.ylabel('Probability')
        plt.title(r'$\mathrm{Histogram\ of\ Test accuracy:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
        plt.grid(True)
        '''
        x = np.arange(1, args.cpasses+1)
        l = plt.plot(x, test_accuracy, 'r', linewidth=2)

        plt.show()
    else:
        print("Give a model to check")