import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import leaky_relu
from torch.nn.utils import spectral_norm

def get_nonspade_norm_layer(args, norm_type='spectralinstance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)
    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
            
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

def custom_model1(in_chan, out_chan):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4,4), stride=2, padding=1)),
        nn.LeakyReLU(inplace=True)
    )

def custom_model2(in_chan, out_chan, stride=2):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4,4), stride=stride, padding=1)),
        nn.InstanceNorm2d(out_chan),
        nn.LeakyReLU(inplace=True)
    )

class SPADEDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer1 = custom_model1(args.total_nc, 64)
        self.layer2 = custom_model2(64, 128)
        self.layer3 = custom_model2(128, 256)
        self.layer4 = custom_model2(256, 512, stride=1)
        self.inst_norm = nn.InstanceNorm2d(512)
        self.conv = spectral_norm(nn.Conv2d(512, 1, kernel_size=(4,4), padding=1))

    def forward(self, img, seg):
        x = torch.cat((seg, img.detach()), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = leaky_relu(self.inst_norm(x))
        x = self.conv(x)
        return x

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        for i in range(args.num_scale):
            subnetD = self.create_single_discriminator(args)
            self.add_module('discriminator_%d' % i, subnetD)
            
            subnetDD = self.create_single_dilateddiscriminator(args)
            self.add_module('discriminator_%d' % int(str(i+1)), subnetDD)

    def create_single_discriminator(self, args):
        netD = NLayerDiscriminator(args)
        return netD

    def create_single_dilateddiscriminator(self, args):
        netD = NLayerDilatedDiscriminator(args)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input):
        result = []
        get_intermediate_features = not self.args.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)

        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        #padw = 0
        nf = args.d_num_filter
        n_first = nf
        input_nc = 20*args.img_nc# + 3 # 10 = 6+2+2
        norm_layer = get_nonspade_norm_layer(args, 'spectralinstance')

        sequence = [[nn.Conv2d(input_nc, n_first, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]#129,129

        nf_prev = n_first
        for n in range(1, args.d_num_layer):
            nf = min(nf * 2, 1024)#66,66 34,34 35,35 / 66,66 67,67 / 130,130
            stride = 1 if n == args.d_num_layer - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]
            nf_prev = nf

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]] #35,35

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.args.no_ganFeat_loss
        if get_intermediate_features:
            #return results
            return results[1:]
        else:
            return results[-1]

class NLayerDilatedDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        #padw = 0
        nf = args.d_num_filter
        n_first = nf
        #n_first = nf
        #input_nc = args.label_nc + args.img_nc
        #input_nc = 9*args.img_nc + 2
        input_nc = 20*args.img_nc# + 3 # 10 = 6+2+2
        norm_layer = get_nonspade_norm_layer(args, 'spectralinstance')

        sequence = [[nn.Conv2d(input_nc, n_first, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]#129,129

        nf_prev = n_first
        for n in range(1, args.d_num_layer):
            nf = min(nf * 2, 1024)#66,66 34,34 35,35 / 66,66 67,67 / 130,130
            stride = 1 if n == args.d_num_layer - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw,dilation=2)),
                          nn.LeakyReLU(0.2, False)
                          ]]
            nf_prev = nf

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]] #35,35

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.args.no_ganFeat_loss
        if get_intermediate_features:
            #return results
            return results[1:]
        else:
            return results[-1]