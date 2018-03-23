import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description='Stochastic BNN')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--test-batch-size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dpath', type=str, default="./datasets/")
parser.add_argument('--log-interval', type=int, default=50)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
trainset = torchvision.datasets.CIFAR10(
    root=args.dpath, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(
    root=args.dpath, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, **kwargs)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


net = Net()
if args.cuda:
    torch.cuda.set_device(0)
    net.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 1):
        # get the inputs
        inputs, labels = data
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if batch_idx % args.log_interval == 0:
            print('[%d, %5d] loss: %.6f' %
                  (epoch, batch_idx, running_loss / 2000.0))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images, volatile=True), Variable(labels, volatile=True)
        output = net(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.data.size(0)
        prediction = output.data.max(1)[1]
        correct += (prediction == labels.data).cpu().sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
