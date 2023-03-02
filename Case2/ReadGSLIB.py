import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Convert():
    def __init__(self,root):
        self.root = root
        file = open(self.root,'r')
        self.file_name = file.readline()
        property_num = file.readline()
        self.startline = int(property_num) + 2
        head = []
        for i in range (int(property_num)):
            property_name = file.readline()
            property_name = property_name.strip('\n')
            head.append(property_name)
        self.head = head
        file.close()

    def get_data(self):
        dim = self.get_dim()
        data = pd.read_csv(self.root,sep=' ',skiprows=self.startline, names=self.head)
        if len(data) != dim[0]*dim[1]*dim[2]:
            dim[2] = int(len(data)/(dim[0]*dim[1]))
        return dim, data[0:dim[0]*dim[1]*dim[2]]

    def get_dim(self):
        dim = re.findall(r"\d+\.?\d*",self.file_name)
        dim = [int(i) for i in dim]
        return dim

def color_map(mode=None):

    if mode == 'facies':
        cmap = mcolors.ListedColormap([(255/255,127/255,0/255),   #channel lag   1
                                        (255/255,255/255,0/255),  #point bar     2
                                        (191/255,191/255,140/255),#sand plug     3
                                        (204/255,127/255,51/255), #cs I          4
                                        (204/255,255/255,51/255), #cs channel    5
                                        (204/255,204/255,51/255), #cs II         6
                                        (102/255,204/255,51/255), #levee         7
                                        (0/255,255/255,0/255),    #overbank      8
                                        (0/255,204/255,127/255)]) #mud plug      9
        #boundaries = [0,1,2,3,4,5,6,7,8,9]
        boundaries = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)

    elif mode == '7':
        cmap = mcolors.ListedColormap([(255/255,127/255,0/255),   #channel lag   1
                                        (255/255,255/255,0/255),  #point bar     2
                                        (191/255,191/255,140/255),#sand plug     3
                                        (204/255,255/255,51/255), #crevass splay 4
                                        (102/255,204/255,51/255), #levee         5
                                        (0/255,255/255,0/255),    #overbank      6
                                        (0/255,204/255,127/255)]) #mud plug      7
        #boundaries = [0,1,2,3,4,5,6,7]
        boundaries = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
        #boundaries = [1,2,3,4,5,6,7,8]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)
    elif mode == '3':
        cmap = mcolors.ListedColormap([(255/255,255/255,0/255),
                                        (191/255,191/255,140/255), 
                                        (0/255,255/255,0/255),])
        #boundaries = [0,1,2]
        boundaries = [-0.5,0.5,1.5,2.5]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)
    
    elif mode == 'levee':
        cmap = mcolors.ListedColormap([(255/255,255/255,0/255),
                                        (102/255,204/255,51/255), 
                                        (0/255,255/255,0/255),])
        #boundaries = [0,1,2]
        boundaries = [-0.5,0.5,1.5,2.5]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)

    elif mode == 'binary':
        cmap = mcolors.ListedColormap([(255/255,255/255,0/255),
                                        (0/255,255/255,0/255),])
        #boundaries = [0,1,2]
        boundaries = [-0.5,0.5,1.5]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)

    elif mode == 'grain_size':
        cmap= mcolors.ListedColormap([(84/255,171/255,43/255),
                                        (120/255,181/255,51/255),
                                        (150/255,191/255,56/255),
                                        (176/255,204/255,66/255),
                                        (199/255,214/255,76/255),
                                        (217/255,219/255,84/255),
                                        (232/255,227/255,87/255),
                                        (237/255,224/255,61/255),
                                        (240/255,224/255,61/255),
                                        (247/255,222/255,31/255),
                                        (255/255,214/255,13/255),
                                        (247/255,176/255,13/255),
                                        (240/255,138/255,10/255),
                                        (237/255,112/255,13/255),
                                        (232/255,82/255,20/255),
                                        (230/255,56/255,23/255)])
        boundaries = [0,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,
                        0.5625,0.625,0.6875,0.75,0.8125,0.875,0.9375,1]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)
    else:
        cmap=None
        norm=None
    return cmap, norm
"""
    if mode == 'facies':
        cmap = mcolors.ListedColormap([(153/255,153/255,229/255),
                                        (255/255,127/255,0/255),
                                        (255/255,255/255,0/255),
                                        (191/255,191/255,140/255),
                                        (204/255,127/255,51/255),
                                        (204/255,255/255,51/255), 
                                        (204/255,204/255,51/255), 
                                        (102/255,204/255,51/255), 
                                        (0/255,255/255,0/255),
                                        (0/255,204/255,127/255),
                                        (127/255,127/255,127/255),
                                        (216/255,114/255,216/255),
                                        (153/255,204/255,229/255),
                                        (255/255,204/255,255/255)])
        #boundaries = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        boundaries = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)


    if mode == 'facies':
        cmap = mcolors.ListedColormap([(255/255,127/255,0/255),
                                        (255/255,255/255,0/255),
                                        (191/255,191/255,140/255),
                                        (204/255,127/255,51/255),
                                        (204/255,255/255,51/255), 
                                        (204/255,204/255,51/255), 
                                        (102/255,204/255,51/255), 
                                        (0/255,255/255,0/255),
                                        (0/255,204/255,127/255)])
        #boundaries = [0,1,2,3,4,5,6,7,8,9]
        boundaries = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
        norm = mcolors.BoundaryNorm(boundaries,cmap.N,clip=True)
"""