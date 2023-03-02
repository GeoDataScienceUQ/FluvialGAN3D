import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../') 
import ReadGSLIB
from models.generator import Generator, SPADEGenerator
import os
cmap,norm = ReadGSLIB.color_map(mode='7')

checkfolder = r'./test'
isExist = os.path.exists(checkfolder)
if not isExist:
    os.makedirs(checkfolder)

# Create class to resemble argparse
class Args:
    def __init__(self, spade_filter=128, spade_kernel=3, spade_resblk_kernel=3, 
                gen_input_size=256, gen_hidden_size=128, g_num_filter=32, 
                img_nc = 1, label_nc=8):
        self.spade_filter = spade_filter
        self.spade_kernel = spade_kernel
        self.spade_resblk_kernel = spade_resblk_kernel
        self.gen_input_size = gen_input_size
        self.gen_hidden_size = gen_hidden_size
        self.g_num_filter = g_num_filter
        self.img_nc = img_nc
        self.label_nc = label_nc

args = Args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise Exception('GPU not available')

gen2d = Generator(args).cuda()
gen3d = SPADEGenerator(args).cuda()
def load_network(net, label, epoch):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    #net.load_state_dict(weights,strict=False)
    return net

def _tolong(input_onehot):
    bs, _, h, w = input_onehot.size()
    out=input_onehot.argmax(1).view([bs,1,h,w])
    return out

def onehot(input_image):
    input_image = input_image.long()
    input_image = input_image.cuda()
    f_map = input_image
    bs, c, h, w = f_map.size()
    nc = 7
    out_label = torch.cuda.FloatTensor(bs, nc*c, h, w).zero_()
    for i in range(c):
        one_map=f_map[::,i,::,::].view([bs,1,h,w])
        #scatter index value < dim
        out_label[::,i*nc:i*nc+nc,::,::].scatter_(1, one_map, 1.0)

    return out_label

gen2d = load_network(gen2d, 'G', '2d')
gen3d = load_network(gen3d, 'G', '50')

gen2d.eval()
gen3d.eval()
### Start simulation
class Simulation():
    def __init__(self, num, Nav, noise=None):
        if noise is None:
            self.noise = torch.randn(1, 128, dtype=torch.float32, device=device)
        else:
            self.noise = noise.cuda()
        self.Facies = []
        self.num = num
        self.Nav = torch.tensor(Nav).reshape(len(self.noise),1,1,1).cuda()
    # Random 2D unconditional realization
    def simulate_2d(self):
        with torch.no_grad():
            fake = gen2d(self.noise)
            fi = _tolong(fake)
            f = fi.detach().cpu().numpy()
            self.Facies.append(f)

        return fi.detach()
    def group_L(self, inputTensor, group1, group2, group3, group4, group5):
        inputTensor = inputTensor.float()
        bs,c,h,w = inputTensor.size()
        out = torch.cuda.FloatTensor(bs,5,h,w).zero_()
        for i in group1:
            out[::,0,::,::] += inputTensor[::,int(i),::,::]
        for i in group2:
            out[::,1,::,::] += inputTensor[::,int(i),::,::]
        for i in group3:
            out[::,2,::,::] += inputTensor[::,int(i),::,::]
        for i in group4:
            out[::,3,::,::] += inputTensor[::,int(i),::,::]
        for i in group5:
            out[::,4,::,::] += inputTensor[::,int(i),::,::]
        return out
    # Simulate 2D cube bottom upwards
    def simulate_3d(self,fake):
        with torch.no_grad():
            seg = onehot(fake)
            fake = gen3d(self.noise, seg)
            fi = _tolong(fake).detach()
            f = fi.cpu().numpy()
            self.Facies.append(f)
            for i in range(2,self.num):
                seg = 0.1*self.Nav*seg + (1.0-0.1*self.Nav)*fake.detach()
                fake = gen3d(self.noise, seg)
                fi = _tolong(fake).detach()
                f = fi.cpu().numpy()
                self.Facies.append(f)

    def run(self):
        fake = self.simulate_2d()
        self.simulate_3d(fake)
        return self.noise, np.array(self.Facies).squeeze()
#2mins/100simulations
for i in tqdm(range(100)):

    Nav = 5 #standard | HAR: Nav = 1 | LAR: Nav = 5
    # Use this option if you need control the latent vector
    #noise = torch.from_numpy(np.load('./test_/Z%d_3d.npy'%i))
    # Else, you can run it randomly
    noise = None
    noise, Facies = Simulation(32,Nav,noise).run()
    np.save('./test/Z%d_3d.npy'%i,noise.detach().cpu().numpy())
    np.save('./test/Npy%d_3d.npy'%i,Facies)
