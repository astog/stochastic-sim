from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from dataloader import KFoldDataset

# Models
# from model_archive.mlp import Net
# from model_archive.lenet import Net
# from model_archive.alexnet import Net
from model_archive.vgg import Net as model
# from sbnn_models.ripple.mlp import Net
# from sbnn_models.ripple.lenet import Net
# from sbnn_models.wave.mlp import Net
# from sbnn_models.wave.lenet import Net

import time
import datetime
import math
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

means = (0.4914, 0.4822, 0.4465)
stds = (0.2023, 0.1994, 0.2010)

# Training settings
parser = argparse.ArgumentParser(description='Main script for CIFAR10 dataset')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--no-shuffle', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--hunits', type=int, default=32)
parser.add_argument('--include-bias', action='store_true', default=False)
parser.add_argument('--npasses', type=int, default=8)
parser.add_argument('--kfolds', type=int, default=5)
parser.add_argument('--dp-dense', type=float, default=0.5)
parser.add_argument('--dp-conv', type=float, default=0.0)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--dpath', type=str, default="./pytorch_data/")
parser.add_argument('--download', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--no-save', action='store_true', default=False)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--check', type=str, default=None)
parser.add_argument('--test-count', type=int, default=100)

args = parser.parse_args()

for arg in vars(args):
    print("{0:{1}<20} {2}".format(str(arg) + ' ', '-', getattr(args, arg)))
print("\n")

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.shuffle = not args.no_shuffle
args.save = not args.no_save

if args.save_path is None:
    args.save_path = "{}-{}.mkl".format("model", int(time.time()))

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
# model = Net(3, 10, args.npasses, bias=args.include_bias)
# model = Net(3, 10, bias=args.include_bias)

if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, nesterov=True, weight_decay=args.wd)
optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,
    amsgrad=False,
    weight_decay=args.wd
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, cooldown=10, verbose=True, min_lr=5e-7)


def train(epoch):
    # Initialize training dataset with folds
    kfold_dataset = KFoldDataset(train_dataset, args.kfolds)

    total_train_loss = 0.0
    total_val_loss = 0.0
    total_val_correct = 0

    # Go through all folds
    for ifold in xrange(args.kfolds):
        print("({}, {})".format(epoch, ifold+1))

        # Get randomized subset
        train_set, val_set = kfold_dataset.get_datasets(ifold)

        # Initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

        print("| Training")
        train_loss = do_train(train_loader, epoch, ifold)
        total_train_loss += train_loss
        scheduler.step(train_loss)

        print("| Validating")
        val_loss, val_correct = do_val(val_loader, epoch, ifold)
        total_val_loss += val_loss
        total_val_correct += val_correct

    print('Validation accuracy: {}/{} ({:.4f}%)\n'.format(
        total_val_correct, len(train_dataset),
        (100. * total_val_correct) / len(train_dataset)
    ))

    return total_train_loss / args.kfolds, total_val_loss / args.kfolds, float(total_val_correct) / len(train_dataset)


def do_train(dataloader, epoch, ifold):
    # Initialize batchnorm and dropout layers for training
    model.train()

    # Logging variables
    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target, requires_grad=False)

        output = model(data)

        loss = criterion(output, target)
        # loss = criterion(output, target) + (args.wd * model.get_regul_loss(mode='pow4'))

        total_loss += loss.data.item()

        model.backward(loss, optimizer)

        if batch_idx % args.log_interval == 0:
            print("| {:3.4f}%\tLoss: {:1.5e}".format(100. * batch_idx / len(dataloader), total_loss), end='\r')

    print("| {:3.4f}%\tLoss: {:1.5e}".format(100., total_loss))
    return total_loss


def do_val(dataloader, epoch, ifold):
    # Initialize batchnorm and dropout layers for testing
    model.train()

    # Logging variabls
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(dataloader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

        output = model(data)

        loss = criterion(output, target).data.item()  # sum up batch loss
        total_loss += loss

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if batch_idx % args.log_interval == 0:
            print("| {:3.4f}%\tLoss: {:1.5e}".format(100. * batch_idx / len(dataloader), total_loss), end='\r')

    print("| {:3.4f}%\tLoss: {:1.5e}".format(100., total_loss), end='\n\n')

    return total_loss, correct


def test(epoch):
    # Initialize batchnorm and dropout layers for testing
    model.eval()

    # Logging variables
    correct = 0

    print("Testing")
    for batch_idx, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data, requires_grad=False)

        output = model(data)

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
    min_val_loss = np.inf
    test_correct = 0
    vt = 0.0
    beta = 0.9

    for epoch in range(1, args.epochs + 1):
        time_start = datetime.datetime.now()

        train_loss, val_loss, val_correct = train(epoch)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            test_correct = test(epoch)
            if args.save:
                torch.save(model.state_dict(), args.save_path)
        else:
            print("Best validation loss was {:.4f}%, got {:.4f}%".format(min_val_loss, val_loss))

        time_complete = datetime.datetime.now() - time_start
        time_complete = time_complete.total_seconds()
        print("\nTime to complete epoch {} == {} sec(s)".format(
            epoch, time_complete
        ))

        # Calculated moving average for time to complete epoch
        vt = (beta * vt) + ((1 - beta) * time_complete)
        average_epoch_time = vt / (1 - pow(beta, epoch))
        print("Estimated time left == {}".format(
            str(datetime.timedelta(seconds=average_epoch_time * (args.epochs - epoch)))
        ))

        print("{:=<72}\n".format(""))

    print('\nFinal Test accuracy: {}/{} ({:.4f}%)'.format(
        test_correct, len(test_loader.dataset),
        100. * (float(test_correct) / len(test_loader.dataset))
    ))
