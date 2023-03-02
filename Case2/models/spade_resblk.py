import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d
from torch.nn.utils import spectral_norm
from .spade import SPADE

class SPADEResBlk(Module):
    def __init__(self, args, fin, fout, use_spade=True):
        super().__init__()

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        kernel_size = args.spade_resblk_kernel
        label_nc = 7
        
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=kernel_size, padding=1)
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=kernel_size, padding=1)
        self.conv_1 = spectral_norm(self.conv_1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.conv_s = spectral_norm(self.conv_s)
            self.norm_s = SPADE(args, fin, label_nc)

        self.norm_0 = SPADE(args, fin, label_nc)
        self.norm_1 = SPADE(args, fmiddle, label_nc)

    def forward(self, x, seg):

        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)