import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from dataloader.geocube3d import Geocube
from models.d3_trainer import D3Trainer
import ReadGSLIB
import os
cmap,norm = ReadGSLIB.color_map(mode='7')

checkfolder = r'./generated'
isExist = os.path.exists(checkfolder)
if not isExist:
    os.makedirs(checkfolder)

def plot(input_tensor,epoch,iteration,num):
    plt.imsave("./generated/Epoch%dIteration%d_%d.png"%(epoch,iteration,num),input_tensor[0][0],vmin=0,vmax=6,cmap=cmap)

path = r'./Dataset/Flumy'
batch_size=8
fold = 'train2d_7f'
dataset = {
    x: Geocube(path, split=x, isSeq=True, isaug=False) for x in [fold]
}

data = {
    x: torch.utils.data.DataLoader(dataset[x], 
                  batch_size=batch_size, 
                  shuffle=True, 
                  num_workers=0,
                  drop_last=True) for x in [fold]
}
# Create class to resemble argparse
class Args:
    def __init__(self, spade_filter=128, spade_kernel=3, spade_resblk_kernel=3,bn=8, 
                gen_input_size=256, gen_hidden_size=128, g_num_filter=32, 
                num_scale=1, d_num_layer=4, d_num_filter=32, img_nc = 1,
                no_ganFeat_loss=True, isTrain=True,
                lr=0.0002,is_continue=False,which_epoch=0,gpu_ids="0"):
        self.spade_filter = spade_filter
        self.spade_kernel = spade_kernel
        self.spade_resblk_kernel = spade_resblk_kernel
        self.bn = bn
        self.gen_input_size = gen_input_size
        self.gen_hidden_size = gen_hidden_size
        self.g_num_filter = g_num_filter
        self.num_scale = num_scale
        self.d_num_layer = d_num_layer
        self.d_num_filter = d_num_filter
        self.img_nc = img_nc
        self.no_ganFeat_loss = no_ganFeat_loss
        self.isTrain = isTrain
        self.is_continue = is_continue
        self.which_epoch = which_epoch
        self.lr = lr
        self.gpu_ids = gpu_ids
        if gen_hidden_size%16 != 0:
            print("Gen hidden size not multiple of 16")

args = Args()
z_dim = args.gen_hidden_size
epochs = 50-args.which_epoch
con = args.which_epoch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise Exception('GPU not available')
torch.backends.cudnn.benchmark = True

def save_loss(filename, value):
    with open (filename,'a') as f:
        f.write(str(value))
        f.write('\n')

trainer = D3Trainer(args)

for epoch in tqdm(range(epochs)):
    for i, data_i in enumerate(data[fold]):
        noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)

        trainer.run_generator_one_step(noise,data_i,num=1)
        trainer.run_discriminator_one_step(noise, data_i,num=1)
        if (i+1)%100 == 0:
            losses = trainer.get_latest_losses()
            save_loss('1g_loss.txt',losses['GAN'].item())
            save_loss('1dr_loss.txt',losses['D_real'].item())
            save_loss('1df_loss.txt',losses['D_Fake'].item())
            save_loss('1gp.txt',losses['D_GP'].item())
        trainer.run_generator_one_step(noise,data_i,num=3)
        trainer.run_discriminator_one_step(noise, data_i,num=3)
        if (i+1)%100 == 0:
            losses = trainer.get_latest_losses()
            save_loss('3g_loss.txt',losses['GAN'].item())
            save_loss('3dr_loss.txt',losses['D_real'].item())
            save_loss('3df_loss.txt',losses['D_Fake'].item())
            save_loss('3gp.txt',losses['D_GP'].item())
        trainer.run_generator_one_step(noise,data_i,num=5)
        trainer.run_discriminator_one_step(noise, data_i,num=5)
        if (i+1)%100 == 0:
            losses = trainer.get_latest_losses()
            save_loss('5g_loss.txt',losses['GAN'].item())
            save_loss('5dr_loss.txt',losses['D_real'].item())
            save_loss('5df_loss.txt',losses['D_Fake'].item())
            save_loss('5gp.txt',losses['D_GP'].item())
        trainer.run_generator_one_step(noise,data_i,num=7)
        trainer.run_discriminator_one_step(noise, data_i,num=7)
        if (i+1)%100 == 0:
            losses = trainer.get_latest_losses()
            save_loss('7g_loss.txt',losses['GAN'].item())
            save_loss('7dr_loss.txt',losses['D_real'].item())
            save_loss('7df_loss.txt',losses['D_Fake'].item())
            save_loss('7gp.txt',losses['D_GP'].item())

        if (i+1)%500 == 0:
            with torch.no_grad():
                fake_img = trainer.get_latest_generated()
                fake_img_np = fake_img.detach().cpu().numpy()
                plot(fake_img_np,epoch+1+con,i+1,7)

                noise = torch.randn(batch_size, z_dim, dtype=torch.float32, device=device)
                generated = trainer.test_generate(noise,data_i)
                fake_img_np = generated.detach().cpu().numpy()
                plot(fake_img_np,epoch+1+con,i+1,1)

                noise = torch.randn(batch_size, z_dim, dtype=torch.float32, device=device)
                generated = trainer.test_generate(noise,data_i,num=3)
                fake_img_np = generated.detach().cpu().numpy()
                plot(fake_img_np,epoch+1+con,i+1,3)

                noise = torch.randn(batch_size, z_dim, dtype=torch.float32, device=device)
                generated = trainer.test_generate(noise,data_i,num=5)
                fake_img_np = generated.detach().cpu().numpy()
                plot(fake_img_np,epoch+1+con,i+1,5)

    if (epoch+1+con)%1 == 0:
        trainer.save(epoch+1+con)