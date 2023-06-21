'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import torchattacks
from models import *
# from utils import progress_bar
from losses import ReluLoss
import pickle


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--budget', default=1.0, type=float, help='relu budget')
parser.add_argument('--device', default='cuda:0', type=str, help='device')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = torch.device(args.device)
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# class CIFAR10MOD(torchvision.datasets.CIFAR10):
#     def __init__(self, *args, **kwargs):
#         return super(CIFAR10MOD, self).__init__(*args, **kwargs)
        
#     def __getitem__(self, index):
#         return super(CIFAR10MOD, self).__getitem__(index), index
    
# trainset = CIFAR10MOD( root='./data', train=True, download=True, transform=transform_train)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
model = torch.load('/home/rhossain/exp/checkpoint/ckpt_relu_modelVGG16_fullModel_budget_1.0.pth')

atk = torchattacks.PGD(model, eps=0.031, alpha=.01, steps=7, random_start=True)
atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

adv_examples = []

for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    adv_data = atk(inputs, targets)
    adv_examples.append((adv_data, targets))
    
# Model
print('==> Building model..')

model_name = 'VGG16'
# net = VGG('VGG16')
net = VGGMod(model_name)

print(net)

# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

""" if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] """

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=240)

loss_relu = ReluLoss(args.budget)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ((inputs, targets), (adv_example,label)) in enumerate(zip(trainloader, adv_examples)):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_data, adv_targets = adv_example.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets) 
        adv_outputs = net(adv_data)
        adv_loss = criterion(adv_outputs, adv_targets)
        print(adv_loss)
        loss1 = loss_relu(net)
        # print(loss1)
        loss = loss + loss1 + adv_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    print(f'Train Loss: {(train_loss/(batch_idx+1)):.3f}, Train Acc: {(100.*correct/total):.3f}')


def test(epoch):
    global best_acc
    global acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        print(f'Test Loss: {test_loss/(batch_idx+1):.3f}, Test Acc: {100.*correct/total:.3f}')

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        #torch.save(net, f'/home/rhossain/exp/checkpoint/ckpt_relu_adversarial_model{model_name}_fullModel_budget_{args.budget}.pth')
        best_acc = acc

# saving in a txt file
#f = open(f'/home/rhossain/exp/checkpoint/ckpt_relu_adversarial_model{model_name}_fullModel_budget_{args.budget}.txt', 'a')
#f.write(f'{net}\n')

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

    total = 0
    relu_applied = 0
    for m in net.modules():
        if isinstance(m, TrainableReLU):
            mask = torch.sigmoid(m.mask)
            relu_applied += sum(mask.ge(0.5)).float()
            total += len(mask)
    print(f'percentage relu: {100 * relu_applied/total:.5f}')
    #f.write(f'epoch: {epoch} percentage relu: {100 * relu_applied/total:.5f} accuracy {acc}\n')
            