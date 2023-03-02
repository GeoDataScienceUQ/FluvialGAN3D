import torch
import torch.nn as nn
from torch.nn import Module, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import interpolate, relu

class SPADE(Module):
    def __init__(self, args, norm_nc, label_nc):
        super().__init__()
        #norm_type = args.norm_type
        num_filters = args.spade_filter
        kernel_size = args.spade_kernel

        pw = kernel_size//2
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, num_filters, kernel_size=(kernel_size, kernel_size), padding=pw),nn.ReLU())
        self.conv_gamma = nn.Conv2d(num_filters, norm_nc, kernel_size=(kernel_size, kernel_size), padding=pw)
        self.conv_beta = nn.Conv2d(num_filters, norm_nc, kernel_size=(kernel_size, kernel_size), padding=pw)

    def forward(self, x, seg):

        normalized = self.param_free_norm(x)
        segmap = interpolate(seg, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out
