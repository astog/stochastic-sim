from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from Adam import Adam

# Models
# from sbnn_models.ripple.mlp import Net
# from model_archive.lenet5 import Net
from sbnn_models.ripple.lenet5 import Net
# from model_archive.mnist_mlp import Net

import time
import datetime
import math
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='Stochastic BNN for MNIST MLP')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lr-start', type=float, default=0.001)
parser.add_argument('--lr-end', type=float, default=0.00001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--no-shuffle', action='store_true', default=False)
parser.add_argument('--valid-pcent', type=float, default=0.15)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hunits', type=int, default=32)
parser.add_argument('--npasses', type=int, default=8)
parser.add_argument('--dp-dense', type=float, default=0.1)
parser.add_argument('--dp-conv', type=float, default=0.0)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--dpath', type=str, default="./pytorch_data/")
parser.add_argument('--download', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--no-save', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=50)
parser.add_argument('--check', type=str, default=None)
parser.add_argument('--test-count', type=int, default=100)

args = parser.parse_args()

for arg in vars(args):
    print("{0:{1}<20} {2}".format(str(arg) + ' ', '-', getattr(args, arg)))
print("\n")

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.shuffle = not args.no_shuffle
args.save = not args.no_save

LR_decay = (args.lr_end / args.lr_start)**(1. / args.epochs)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Dataset transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Cuda dataset arguments
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = datasets.MNIST(args.dpath, train=True, download=args.download, transform=transform)
valid_dataset = datasets.MNIST(args.dpath, train=True, download=args.download, transform=transform)
test_dataset = datasets.MNIST(args.dpath, train=False, download=args.download, transform=transform)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(args.valid_pcent * num_train))

if args.shuffle:
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)


# model = Net(28 * 28, 10, args.hunits, args.npasses, bias=False, dp_dense=args.dp_dense)
# model = Net()
model = Net(args.npasses, bias=False)
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr_start, amsgrad=True)


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

        output = model(data)

        loss = criterion(output, target)
        model.backward(loss, optimizer)

        train_batch_count += 1
        train_batch_avg_loss += float(loss)
        train_batch_avg_count += 1

        if batch_idx % args.log_interval == 0:
            print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
                epoch, 100. * train_batch_count / len(train_loader),
                train_batch_avg_loss / train_batch_avg_count
            ))
            train_batch_avg_loss = 0
            train_batch_avg_count = 0

    if train_batch_avg_count > 0:
        print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
            epoch, 100. * train_batch_count / len(train_loader),
            train_batch_avg_loss / train_batch_avg_count
        ))


def validate(epoch):
    # Initialize batchnorm and dropout layers for testing
    model.eval()

    # Logging variables
    correct = 0
    valid_batch_count = 0
    valid_batch_avg_loss = 0
    valid_batch_avg_count = 0
    val_loss = 0

    for batch_idx, (data, target) in enumerate(valid_loader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        loss = criterion(output, target).data[0]  # sum up batch loss

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        valid_batch_count += 1
        valid_batch_avg_loss += float(loss)
        val_loss += float(loss)
        valid_batch_avg_count += 1

        if batch_idx % args.log_interval == 0:
            print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
                epoch, 100. * valid_batch_count / len(valid_loader),
                valid_batch_avg_loss / valid_batch_avg_count
            ))
            valid_batch_avg_loss = 0
            valid_batch_avg_count = 0

    if valid_batch_avg_count > 0:
        print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
            epoch, 100. * valid_batch_count / len(valid_loader),
            valid_batch_avg_loss / valid_batch_avg_count
        ))

    print('\nValidation set accuracy: {}/{} ({:.4f}%)'.format(
        correct, len(valid_loader) * args.batch_size,
        100. * (float(correct) / (len(valid_loader) * args.batch_size))
    ))

    return correct, val_loss


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
            print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
                epoch, 100. * test_batch_count / len(test_loader),
                test_batch_avg_loss / test_batch_avg_count
            ))
            test_batch_avg_loss = 0
            test_batch_avg_count = 0

    if test_batch_avg_count > 0:
        print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
            epoch, 100. * test_batch_count / len(test_loader),
            test_batch_avg_loss / test_batch_avg_count
        ))

    print('\nTest set accuracy: {}/{} ({:.4f}%)'.format(
        correct, len(test_loader) * args.batch_size,
        100. * (float(correct) / (len(test_loader) * args.batch_size))
    ))

    return correct


def adam_set_lr(optimizer, epoch):
    for param in optimizer.param_groups:
        param['lr'] = args.lr_start / pow(epoch, 1.0/4.0)


if __name__ == '__main__':
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(valid_loader))
    print("Test batches:", len(test_loader), end='\n\n')
    save_model_path = "{}-{}.mkl".format("model", int(time.time()))

    if args.check is not None:
        print("Loading module", args.check)
        model.load_state_dict(torch.load(args.check))

        test_accuracy = np.zeros(args.test_count)
        for itest in xrange(args.test_count):
            test_accuracy[itest] = (float(test(itest)) / (len(test_loader) * args.batch_size))

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

        plt.show()
    else:
        max_valid_correct = 0
        test_correct = 0
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_decay)
        for epoch in range(1, args.epochs + 1):
            time_start = time.clock()
            train(epoch)
            # scheduler.step(epoch=epoch)
            adam_set_lr(optimizer, epoch)
            print("\n{:-<72}".format(""))
            print("Validation:\n")
            correct, val_loss = validate(epoch)
            if correct > max_valid_correct:
                max_valid_correct = correct
                print("\n{:-<72}".format(""))
                print("Test:\n")
                test_correct = test(epoch)
                if args.save:
                    torch.save(model.state_dict(), save_model_path)
            else:
                print('\nBest Validation set accuracy: {}/{} ({:.4f}%)'.format(
                    max_valid_correct, len(valid_loader) * args.batch_size,
                    100. * (float(max_valid_correct) / (len(valid_loader) * args.batch_size))
                ))

            time_complete = time.clock() - time_start
            print("\nTime to complete epoch {} == {} sec(s)".format(
                epoch, time_complete
            ))
            print("Estimated time left == {}".format(
                str(datetime.timedelta(seconds=time_complete * (args.epochs - epoch)))
            ))

            print("{:=<72}\n".format(""))

        print('\nFinal Test set accuracy: {}/{} ({:.4f}%)'.format(
            test_correct, len(test_loader) * args.batch_size,
            100. * (float(test_correct) / (len(test_loader) * args.batch_size))
        ))
