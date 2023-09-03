import torch
import torch.nn as nn
# import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group
#from utils.options import args

DEFAULT_THRESHOLD = 5e-3

def _gumbel_sigmoid(input, temperature=1, hard=False, eps = 1e-10):
    """
    A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
    In short, it's a function that mimics #[a>0] indicator where a is the logit
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    
    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    
    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
    
    
    :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
    :param eps: a small number used for numerical stability
    :returns: a callable that can (and should) be used as a nonlinearity
    """
    # @staticmethod
    # def forward(ctx, input, temperature=1, hard=False, eps = 1e-10):
    with torch.no_grad():
        # generate a random sample from the uniform distribution
        uniform1 = torch.rand(input.size())
        uniform2 = torch.rand(input.size())
        gumbel_noise = -torch.log(torch.log(uniform1 + eps)/torch.log(uniform2 + eps) + eps).cuda()

    reparam = (input + gumbel_noise)/temperature
#         print(reparam)
    y_soft = torch.sigmoid(reparam)     
    if hard:
        # Straight through.
        index = (y_soft > 0.5).nonzero(as_tuple=True)[0] 
        y_hard = torch.zeros_like(input, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

""" class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step,th):
        ctx.step= step.item()
        ctx.th = th.item()         
        output = input.clone().zero_()
        
        output[input.ge(ctx.th)] = 1
        output[input.le(ctx.th)] = -1
        
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        
        return grad_input, None,None
                
quantize1 = _Quantize.apply """

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)

        self.Beta=torch.Tensor(torch.ones(self.weight.size()[1])).fill_(1).cuda()
        self.mask=nn.Parameter(torch.Tensor(torch.ones(self.weight.size()[1])).fill_(1), requires_grad=True)
        # self.B_mask.fill_(0)

        
    def forward(self, input):
       
        self.N_bits = 1
        #th = self.B_mask.mean()#self.weight.mean()
        
        #step = self.B_mask[self.B_mask.ge(th)+self.B_mask.le(-th)].abs().mean()
        
        self.masked_bw = _gumbel_sigmoid(self.mask*self.Beta, hard=True)
        #print(self.masked_bw.shape)
        masked_weight = self.masked_bw #[:, None]#, None, None]
        #print(masked_weight.shape)

        return F.linear(input, self.weight*masked_weight , self.bias)



class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()
        self.epoch = -1

        w = self.weight
        self.a, self.b = get_ab(np.prod(w.shape[1:]))
        R1 = torch.tensor(ortho_group.rvs(dim=self.a)).float().cuda()
        R2 = torch.tensor(ortho_group.rvs(dim=self.b)).float().cuda()
        self.register_buffer('R1', R1)
        self.register_buffer('R2', R2)
        self.Rweight = torch.ones_like(w)

        ## Li: q, beta,mask and counter variables
        # self.q_val=nn.Parameter(torch.Tensor(torch.ones(self.weight.size()[0])).fill_(1)).cuda()
        self.Beta=torch.Tensor(torch.ones(self.weight.size()[0])).fill_(1).cuda()
        self.mask=nn.Parameter(torch.Tensor(torch.ones(self.weight.size()[0])).fill_(1), requires_grad=True) ## Original continous mask S
        # self.mask.data[0:int(2*(self.mask.size()[0])/3)] = 0.01 ## setting first 1/3 of them to 1
        self.mask.data[int((self.mask.size()[0])/3):] =  torch.zeros(self.mask.data[int((self.mask.size()[0])/3):].size()).random_(-10,-1).div_(10) 

        self.masked_bw = torch.Tensor(torch.ones(self.weight.size()[0])).fill_(0)
        self.masked_bw[:int((self.mask.size()[0])/3)] = 1

        self.temperature = 1
        self.threshold = DEFAULT_THRESHOLD
        self.searching = True
        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1).detach()
        self.alpha = nn.Parameter(sw.cuda(), requires_grad=True)
        self.rotate = nn.Parameter(torch.ones(w.size(0), 1, 1, 1).cuda()*np.pi/2, requires_grad=True)
        self.Rotate = torch.zeros(1)

    def forward(self, input, indicator=1):
        a0 = input
      
        w = self.weight
        # w1 = w - w.mean([1,2,3], keepdim=True)
        # w2 = w1 / w1.std([1,2,3], keepdim=True)
        # a1 = a0 - a0.mean([1,2,3], keepdim=True)
        # a2 = a1 / a1.std([1,2,3], keepdim=True)
        # a, b = self.a, self.b
        # X = w2.view(w.shape[0], a, b)
        # if self.epoch > -1 and self.epoch % args.rotation_update == 0:
        #     for _ in range(3):
        #         #* update B
        #         V = self.R1.t() @ X.detach() @ self.R2
        #         B = torch.sign(V)
        #         #* update R1
        #         D1 = sum([Bi@(self.R2.t())@(Xi.t()) for (Bi,Xi) in zip(B,X.detach())])
        #         U1, S1, V1 = torch.svd(D1)
        #         self.R1 = (V1@(U1.t()))
        #         #* update R2
        #         D2 = sum([(Xi.t())@self.R1@Bi for (Xi,Bi) in zip(X.detach(),B)])
        #         U2, S2, V2 = torch.svd(D2)
        #         self.R2 = (U2@(V2.t()))
        # self.Rweight = ((self.R1.t())@X@(self.R2)).view_as(w)
        # delta = self.Rweight.detach() - w2
        # w3 = w2 + torch.abs(torch.sin(self.rotate)) * delta

        ### Li: performing the forward operation
        #bw = BinaryQuantize().apply(w3, self.k.to(w.device), self.t.to(w.device))

        #if self.searching:
            #print(self.mask.requires_grad)
            
        # Adnan, this is new gumbel sigmoid method
        self.masked_bw = _gumbel_sigmoid(self.mask*self.Beta, hard=True)
        masked_weight = self.masked_bw[:, None, None, None]

            
            # mask_sig = torch.sigmoid((self.mask.cuda()-self.threshold)*self.Beta.cuda()).flatten()
            # gumbel_input = torch.cat((mask_sig, 1-mask_sig),0).view(2, mask_sig.size(0)).transpose(1,0)
            # one_hot = F.gumbel_softmax(torch.log(gumbel_input+1e-10), tau=self.temperature, hard=True)
            # self.masked_bw.data = one_hot[:,0]
            # masked_weight = self.masked_bw[:, None, None, None]

            # this is ICLR'21
            # masks = torch.sigmoid(self.mask * self.Beta)
            # self.q_val = torch.bernoulli(masks)
            # self.masked_bw = masks * self.q_val 
            #bw= w * masked_weight
            
        """  else:
            # if fix the mask, only train weight, do this
            if self.bias is None:
                bias = None
            else:
                bias = self.bias * self.masked_bw.detach().cuda()
            masked_weight = self.masked_bw[:, None, None, None].detach().cuda()
            with torch.no_grad():
                bw= bw * masked_weight """

        # if input.size()[1]==3:
        #     ba = a2
        # else:
        #     ba = BinaryQuantize_a().apply(a2, self.k.to(w.device), self.t.to(w.device))

        #* 1bit conv
        
        output = F.conv2d(a0,  w*masked_weight, self.bias, self.stride, self.padding,
                            self.dilation, self.groups) #layerwise/elelmet
        
        bias = nn.Parameter(torch.Tensor(torch.ones(output.size())).fill_(1), requires_grad=True).cuda()
        relu = nn.ReLU()
        bias = relu(bias)
        #* scaling factor
        #output = output  * bw#self.alpha
        return output + bias



""" class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = torch.tensor(1.).to(input.device)
        t = max(t, torch.tensor(1.).to(input.device))
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None """


def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i




# this is base
# base + trainable bias across channels
# base + trainable bias across elements 