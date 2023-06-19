import torchattacks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
model = torch.load('/home/rhossain/exp/checkpoint/ckpt_relu_modelVGG16_fullModel_budget_0.01.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print('==> Building model..')


if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
print(model)
atk = torchattacks.PGD(model, eps=3/255, alpha=2/225, steps=3, random_start=True)
atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

def attack():
    print(model)
    model.eval()
    correct = 0
    adv_examples = []
    total = 0
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        adv_data = atk(data, target)
        output = model(adv_data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()          

    acc = 100.*correct/total
    print(f'Test Accuracy {acc:.3f}')
    print('Saving..')
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    torch.save(model, f'/home/rhossain/exp/checkpoint/ckpt_adverserial_attack_budget_0.01.pth')
    # saving in a txt file
    with open('/home/rhossain/exp/checkpoint/ckpt_adverserial_attack_budget_0.01.txt', 'a') as f:
        f.write(f'eplison: 4/255 , step: 4, alpha: 2/255\n')
        f.write(f'Accuracy of the network on the 10000 test images: %.3f %%\n' % (acc))

attack()