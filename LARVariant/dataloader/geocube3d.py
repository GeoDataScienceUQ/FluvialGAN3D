import os
import numpy as np
import torch
from torch.utils.data import Dataset

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)]

def seperate_cube(input_path, isSeq = True, isaug = False):
    output = []
    for i in range(len(input_path)):
        cube_path=input_path[i]
        NO = os.path.basename(cube_path)[:-11]
        num = int(NO[4:])
        if num >=10 and num+60<640:
            if isaug:
                for index in range(0,8):
                    output.append([input_path[i],index])
            else:
                output.append([input_path[i],0])

    return output

def scale_label(input_label,vmin=6,vmax=11):
    d = vmax - vmin
    output_label1 = (input_label - vmin)/d #shale
    output_label2 = (vmax - input_label)/d #sand
    output_label = np.concatenate((output_label1,output_label2))
    return output_label

class Geocube(Dataset):
    def __init__(self, 
                 root, 
                 split='train', 
                 isSeq = False,
                 isaug = False,
                 cube_size=(1, 256, 256)):
        self.root = root
        self.split = split
        self.isSeq = isSeq
        self.isaug = isaug
        self.cube_size = cube_size
        self.files = {}
        self.data = {}
        self.cube_base = os.path.join(self.root, "img", self.split)
        self.imp_base = os.path.join(self.root, "img", self.split)
        self.files[split] = recursive_glob(rootdir=self.cube_base, suffix=".npy")
        if self.isSeq:
            self.data[split] = seperate_cube(self.files[split],isaug=self.isaug)

        print("Found %d %s images" % (len(self.files[split]), split))
        print("%d %s data"% (len(self.data[split]), split))
 
    def __len__(self):
        return len(self.data[self.split])
    
    def data_aug(self,image,index_aug):
        if index_aug == 1:#左转90
            image = np.rot90(image,k=1,axes=(1,2))
            image = np.ascontiguousarray(image)
        elif index_aug == 2:#左转180
            image = np.rot90(image,k=2,axes=(1,2))
            image = np.ascontiguousarray(image)
        elif index_aug == 3:#右转90
            image = np.rot90(image,k=-1,axes=(1,2))
            image = np.ascontiguousarray(image)
        elif index_aug == 4:#上下
            image = np.flip(image,axis=1)
            image = np.ascontiguousarray(image)
        elif index_aug == 5:#左右
            image = np.flip(image,axis=2)
            image = np.ascontiguousarray(image)
        elif index_aug == 6:#负对角线
            image = np.rot90(image,k=1,axes=(1,2))
            image = np.flip(image,axis=1)
            image = np.ascontiguousarray(image)
        elif index_aug == 7:#正对角线
            image = np.rot90(image,k=-1,axes=(1,2))       
            image = np.flip(image,axis=1)
            image = np.ascontiguousarray(image)
        return image

    def __getitem__(self, index):
        index_aug = self.data[self.split][index][1]
        cube_path = self.data[self.split][index][0].rstrip()
        #print(cube_path)
        NO = os.path.basename(cube_path)[:-11]
        Nav = 5
        no = NO[:4]
        num = int(NO[4:])
        lbl_path = os.path.join(
            self.imp_base,
            str(no) + str(int(num-10)) + "_Facies.npy"
        )
        up2_path = os.path.join(
            self.imp_base,
            str(no) + str(int(num+10)) + "_Facies.npy"
        )
        up3_path = os.path.join(
            self.imp_base,
            str(no) + str(int(num+20)) + "_Facies.npy"
        )
        up4_path = os.path.join(
            self.imp_base,
            str(no) + str(int(num+30)) + "_Facies.npy"
        )
        up5_path = os.path.join(
            self.imp_base,
            str(no) + str(int(num+40)) + "_Facies.npy"
        )
        up6_path = os.path.join(
            self.imp_base,
            str(no) + str(int(num+50)) + "_Facies.npy"
        )
        up7_path = os.path.join(
            self.imp_base,
            str(no) + str(int(num+60)) + "_Facies.npy"
        )
        label = np.load(lbl_path).reshape([1,256,256])
        image = np.load(cube_path).reshape([1,256,256])
        up2 = np.load(up2_path).reshape([1,256,256])
        up3 = np.load(up3_path).reshape([1,256,256])
        up4 = np.load(up4_path).reshape([1,256,256])
        up5 = np.load(up5_path).reshape([1,256,256])
        up6 = np.load(up6_path).reshape([1,256,256])
        up7 = np.load(up7_path).reshape([1,256,256])

        label_bottom = torch.from_numpy(label).long().view([1,256,256])
        label_onehot = torch.FloatTensor(7, 256, 256).zero_()
        label_tensor = label_onehot.scatter_(0, label_bottom, 1.0)

        image_tensor = torch.from_numpy(image).view([1,256,256]).float()

        up2_tensor = torch.from_numpy(up2).view([1,256,256]).float()

        up3_tensor = torch.from_numpy(up3).view([1,256,256]).float()
        
        up4_tensor = torch.from_numpy(up4).view([1,256,256]).float()

        up5_tensor = torch.from_numpy(up5).view([1,256,256]).float()

        up6_tensor = torch.from_numpy(up6).view([1,256,256]).float()

        up7_tensor = torch.from_numpy(up7).view([1,256,256]).float()    

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'up2': up2_tensor,
                      'up3': up3_tensor,
                      'up4': up4_tensor,
                      'up5': up5_tensor,
                      'up6': up6_tensor,
                      'up7': up7_tensor,
                      'No':no,
                      'Nav':Nav,
                      'index':num,
                      'index_aug':index_aug,
                      }

        return input_dict