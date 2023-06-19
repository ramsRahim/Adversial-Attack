'''VGG11/13/16/19 in Pytorch.'''
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class TrainableReLU(nn.Module):
    def __init__(self, out_channels, mask_type='channel', threshold=0.5):
        super(TrainableReLU, self).__init__()
        rand_nums = (-1 - 1) * torch.rand(out_channels) + 1
        # rand_nums = torch.ones(out_channels)
        self.mask = nn.Parameter(torch.Tensor(rand_nums), requires_grad=True)
        self.threshold = threshold 

    def forward(self, x):
        mask = torch.sigmoid(self.mask)
        relu_idx = mask.ge(self.threshold)
        # print(mask.shape, x.shape)
        # print(float(sum(relu_idx))/len(mask))
        # print(x[:,relu_idx].shape, x.shape)
        # print(np.prod(x[:,relu_idx].shape[1:],initial=1))
        x[:,relu_idx] = F.relu(x[:,relu_idx])
        return x


class VGGMod(nn.Module):
    def __init__(self, vgg_name, mask_type='channel'):
        super(VGGMod, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.mask_type = mask_type

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           TrainableReLU(x)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
class TrainableReLUPixelWise(nn.Module):
    def __init__(self, mask_shape, threshold=0.5):
        super(TrainableReLUPixelWise, self).__init__()
        rand_nums = (-1 - 1) * torch.rand(*mask_shape) + 1
        # rand_nums = torch.ones(out_channels)
        self.mask = nn.Parameter(torch.Tensor(rand_nums), requires_grad=True)
        self.threshold = threshold 

    def forward(self, x):
        mask = torch.sigmoid(self.mask)
        relu_idx = mask.ge(self.threshold)
        # print(mask.shape, x.shape, relu_idx.shape)
        # print(float(sum(relu_idx))/len(mask))
        # print(x[:,relu_idx].shape, x.shape)
        # print(np.prod(x[:,relu_idx].shape[1:],initial=1))
        x[:,relu_idx] = F.relu(x[:,relu_idx])
        return x


class VGGModPixelWise(nn.Module):
    def __init__(self, vgg_name, mask_type='pixelwise'):
        super(VGGModPixelWise, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.mask_type = mask_type

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @torch.no_grad()
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        inp = torch.rand(1,3,32,32)
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                inp = layers[-1](inp)
            else:
                conv_layer = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                inp = conv_layer(inp)
                mask_shape = inp.shape[1:]
                layers += [conv_layer,
                           nn.BatchNorm2d(x),
                           TrainableReLUPixelWise(mask_shape)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
