import numpy as np
import xarray as xr
from tqdm import tqdm
import os
# test1 = xr.open_dataset('ERA5_plevel_2021_10_01_23.nc')
# test1 = np.array(test1.to_array().squeeze())[:,:]

# fre = np.zeros(test1.shape)
# for var in range(test1.shape[0]):
#     for level in range(test1.shape[1]):
#         test1[var,level] = test1[var,level]/(abs(test1[var,level]).max())

def fre_pixel(x,y,pic):
    #left
    if x==0:
        left_fre = 0
    else:
        if y==0:
            left_fre = ((pic[...,x,y]-pic[...,x-1,y])+(pic[...,x,y]-pic[...,x-1,y+1]))/2
        elif y==pic.shape[-1]-1:
            left_fre = ((pic[...,x,y]-pic[...,x-1,y-1])+(pic[...,x,y]-pic[...,x-1,y]))/2
        else:
            left_fre = ((pic[...,x,y]-pic[...,x-1,y-1])+(pic[...,x,y]-pic[...,x-1,y])+(pic[...,x,y]-pic[...,x-1,y+1]))/3
    #up
    if y==0:
        up_fre = 0
    else:
        if x==0:
            up_fre = ((pic[...,x,y]-pic[...,x,y-1])+(pic[...,x,y]-pic[...,x+1,y-1]))/2
        elif x==pic.shape[-2]-1:
            up_fre = ((pic[...,x,y]-pic[...,x,y-1])+(pic[...,x,y]-pic[...,x-1,y-1]))/2
        else:
            up_fre = ((pic[...,x,y]-pic[...,x-1,y-1])+(pic[...,x,y]-pic[...,x,y-1])+(pic[...,x,y]-pic[...,x+1,y-1]))/3
    #right 
    if x==(pic.shape[-2]-1):
        right_fre = 0
    else:
        if y==0:
            right_fre = ((pic[...,x,y]-pic[...,x+1,y])+(pic[...,x,y]-pic[...,x+1,y+1]))/2
        elif y==pic.shape[-1]-1:
            right_fre = ((pic[...,x,y]-pic[...,x+1,y-1])+(pic[...,x,y]-pic[...,x+1,y]))/2
        else:
            right_fre = ((pic[...,x,y]-pic[...,x+1,y-1])+(pic[...,x,y]-pic[...,x+1,y])+(pic[...,x,y]-pic[...,x+1,y+1]))/3
    #down
    if y==(pic.shape[-1]-1):
        down_fre = 0
    else:
        if x==0:
            down_fre = ((pic[...,x,y]-pic[...,x,y+1])+(pic[...,x,y]-pic[...,x+1,y+1]))/2
        elif x==pic.shape[-2]-1:
            down_fre = ((pic[...,x,y]-pic[...,x,y+1])+(pic[...,x,y]-pic[...,x-1,y+1]))/2
        else:
            down_fre = ((pic[...,x,y]-pic[...,x-1,y+1])+(pic[...,x,y]-pic[...,x,y+1])+(pic[...,x,y]-pic[...,x+1,y+1]))/3
    Fre = (left_fre+up_fre+right_fre+down_fre)
    return abs(Fre)
# if os.path.exists('testfre.npy'):
#     print('读取结果')
#     fre = np.load('testfre.npy')
# else:
#     for index1 in tqdm(range(test1.shape[-2])):
#         for index2 in range(test1.shape[-1]):
#             fre[...,index1,index2] = fre_pixel(index1,index2,test1)
#     print('保存结果')
#     np.save('testfre.npy',fre)
# for var in range(fre.shape[0]):
#     for level in range(fre.shape[1]):
#         fre[var,level] = (fre[var,level]/(abs(fre[var,level]).max()))

# fre = fre[1,15]
# # fre = np.where(fre>0.1,fre,0)

# import seaborn as sns
# import matplotlib.pyplot as plt

# # fig = plt.figure(figsize=(8, 6))
# sns.set_context({"figure.figsize":(16,8)})
# p1 = sns.heatmap(data=fre,cmap="summer",vmin=0)
# s1 = p1.get_figure()
# s1.savefig('heatfigure.jpg',dpi=300)
