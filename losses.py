import torch 
from torch import nn
from torch.nn import functional as F
#from resnet import TrainableReLU
from models.vgg import TrainableReLU

class ReluLoss(nn.Module):
    def __init__(self, budget=0.5, threshold=0.5):
        super(ReluLoss, self).__init__()
        self.budget = budget
        self.threshold = threshold

    def forward(self, net):
        loss = 0
        for m in net.modules():
            if isinstance(m, TrainableReLU):
                # mask = torch.sigmoid(m.mask)
                # loss += mask.ge(self.threshold).float().sum() / len(mask)
                # loss += torch.abs(torch.sigmoid(m.mask).mean() - 0.5)
                #print('dhukse')
                loss += torch.abs(torch.quantile(torch.sigmoid(m.mask), 1.0-self.budget) - self.threshold)
        return loss
    
class ReluLossNew(nn.Module):
    def __init__(self, budget=0.5, threshold=0.5):
        super(ReluLossNew, self).__init__()
        self.budget = budget
        self.threshold = threshold

    def forward(self, net):
        loss = 0
        all_mask_vals = []
        for m in net.modules():
            if isinstance(m, TrainableReLU):
                all_mask_vals.append(torch.sigmoid(m.mask))
                
        all_mask_vals = torch.cat(all_mask_vals)
        # print(all_mask_vals.shape)
        loss += torch.abs(torch.quantile(all_mask_vals, 1.0-self.budget) - self.threshold)
        return loss