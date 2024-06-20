import torch
from torch import nn
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time
from Pyramid5 import pyramid
import shutil
from scipy import sparse
import pickle
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
#一年气候态计算k
#进行3sigma归一化

# 指定seed
# m_seed = random.randint(0, 100)
# m_seed = 92 #22255

# # 设置seed
# torch.manual_seed(m_seed)
# torch.cuda.manual_seed_all(m_seed)
savingpath = 'pth_test/'
folder = os.path.exists(savingpath)
data_dir = 'ERA5_plevel_2021_10_01_23.nc'

start_time = time.time()
# origin_data = np.load('data_pre/ERA5test3.npy')[:,:36,:720,:]
origin_data = xr.open_dataset(data_dir)

origin_data = np.array(origin_data.to_array()).squeeze()
origin_data = origin_data[:,:36,:720]
#origin_data 原始数据 
#im_global 减去气候态的原数据
#im_global_norm 标准化的im_global
#im_global_norm_train 去异常值后的im_global_norm
climatology = np.load('climatgy.npy')[:,:36,:720] #气候态
im_global = origin_data[:,:36,:720] - climatology #减气候态
# im_global = origin_data
im_global_norm=np.ones(im_global.shape)
std__norm = np.ones((im_global.shape[0],im_global.shape[1]))
max_global = np.ones((im_global.shape[0],im_global.shape[1]))
mean_global = np.ones((im_global.shape[0],im_global.shape[1]))
std_global = np.ones((im_global.shape[0],im_global.shape[1]))
print('数据归一化')
for var in tqdm(range(im_global.shape[0])): #最大最小归一化
    for plevel in range(im_global.shape[1]):
        max_global[var,plevel] = abs(im_global[var,plevel]).max()
        im_global_norm[var,plevel,...] = im_global[var,plevel]/max_global[var,plevel]
        mean_global[var,plevel] = im_global[var,plevel].mean()  #abs(im_global[var,plevel]).max()
        std_global[var,plevel] = np.std(im_global[var,plevel])
        # im_global_norm[var,plevel,...] = im_global_norm[var,plevel,...]/(3*np.std(im_global_norm[var,plevel,...]))
        std__norm[var,plevel]=np.std(im_global_norm[var,plevel,...])

print('数据归一化完成，开始压缩')

np.savez(savingpath+'/absmax.npz',absmax=max_global)
im_global_norm_train = im_global_norm

im_final = torch.ones(im_global.shape, dtype=torch.float32)#.float()
acc_top = 2e-2
acc_L4 = 1.2e-2
acc_L2 = 1e-3
optimsig = "adam"
split_coef = 2
k=1
ki=k
kj=ki
kk=ki
unit1 = int(im_global.shape[2]/ki)
unit2 = int(im_global.shape[3]/kj)
unit3 = int(im_global.shape[1]/kk)
device = torch.device('cuda')
parallel = False #True   # True or False 并行化
im_global_norm_train = torch.tensor(im_global_norm_train, dtype=torch.float32)

#输入进INR
succeed = True #False
prev_path = savingpath
for Nchunk in range(ki*kj):
    print('开始压缩第',Nchunk+1,'个分区')
    iidx,iidy = np.unravel_index(Nchunk,(ki,kj))
    im = im_global_norm_train[:,:,iidx*unit1:(iidx+1)*unit1,iidy*unit2:(iidy+1)*unit2] #
    img_L2 = pyramid(savingpath,im,Nchunk,acc_top,acc_L4,acc_L2,optimsig,device,parallel,succeed,prev_path)
    im_final[:,:,iidx*unit1:(iidx+1)*unit1,iidy*unit2:(iidy+1)*unit2] = img_L2

end_time=time.time()
print('总耗时',end_time-start_time,'s')
# im_global=np.array(im_global)
# orgin_data=np.load('data_pre/norm2.npy')[:,:36,:720,:]

data=np.array(im_final)
print('直接替换误差过大值')
MAe_P=(im_global_norm_train-data) #/im_global_norm_train
print(abs(MAe_P).max(),abs(MAe_P).min())
MAe_P=np.array(MAe_P)
sub = np.where(abs(MAe_P)<0.5, 0, im_global_norm_train)
data=np.where(abs(MAe_P)<0.5, data, im_global_norm_train)
# im_modified = Mae+save_file
sub = sub.reshape(sub.shape[0]*sub.shape[1]*sub.shape[2],sub.shape[3])
allmatrix_sp=sparse.csr_matrix(sub)
res_path = savingpath + '/res_scho.pkl'
dic = {'res_var':allmatrix_sp}
file = open(res_path, 'wb')
pickle.dump(dic, file)#转换成稀疏矩阵存储，会小很多

#反标准化
for var in range(im_global.shape[0]): #最大最小归一化
    for plevel in range(im_global.shape[1]):
        data[var,plevel,...] = data[var,plevel]*max_global[var,plevel]

data = np.array(data)+climatology

real_mae = (abs(origin_data-data)).mean()
real_rmse = (((origin_data-data)**2).mean())**0.5
z500_maxerror = abs((origin_data-data))[0,15].max()
z500_meanerror = abs((origin_data-data))[0,15].mean()
z500_rmse = (((origin_data[0,15]-data[0,15])**2).mean())**0.5
print('反归一化后的rmse为',real_rmse,',MAE为',real_mae,',z500的最大误差为',z500_maxerror,',z500的平均误差为',z500_meanerror,',z500的RMSE为',z500_rmse)

with torch.no_grad():
    fig, axs = plt.subplots(2, figsize=(8, 4))
    # for var in range(1):
    var = 1
    k = 15
    axs[0].imshow(data[var,k], cmap='viridis')
    axs[1].imshow(origin_data[var,k], cmap='viridis')
plt.tight_layout()
plt.savefig('output_image.png', dpi = 1000)