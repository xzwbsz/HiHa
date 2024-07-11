import torch
from torch import nn
import numpy as np
import cmaps
import os
import random
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
from Pyramid2 import pyramid,pyramid_uncompress
import shutil
from scipy import sparse
import pickle

start_time = time.time()
im_final = torch.ones((5,36,720,1440))
k=8
ki=k
kj=ki
unit1 = int(im_final.shape[2]/ki)
unit2 = int(im_final.shape[3]/kj)
device = torch.device('cuda')
parallel = True
pthfile = 'meeting'
climatgy = np.load('data_pre/climatgy.npy')[:,:36,:720,:]
mean_std_global = np.load(pthfile+'/absmax.npz')
# std_norm = np.load(pthfile+'/std_norm_4.npy')

for Nchunk in range(ki*kj):
    print('开始解压缩第',Nchunk+1,'个分区')
    iidx,iidy = np.unravel_index(Nchunk,(ki,kj))
    im_final[:,:,iidx*unit1:(iidx+1)*unit1,iidy*unit2:(iidy+1)*unit2]=pyramid_uncompress(pthfile,im_final.shape[1],im_final.shape[2]/ki,im_final.shape[3]/kj,Nchunk,device,parallel)

im_final = np.array(im_final)
file1 = pthfile+'/res_perserve.pkl'
if os.path.exists(file1):
    sub = (pickle.load(open(file1,'rb'))['res_var']).toarray()
    sub = sub.reshape(im_final.shape)
    im_final = np.where(sub!=0,sub,im_final)
    print('替换1完成')
for var in range(im_final.shape[0]): #最大最小归一化
    for plevel in range(im_final.shape[1]):
        # im_final[var,plevel,...] = im_final[var,plevel]*mean_std_global['std'][var,plevel]+mean_std_global['mean'][var,plevel]
        im_final[var,plevel,...] = im_final[var,plevel]*mean_std_global['absmax'][var,plevel]

im_final = im_final + climatgy

file = pthfile+'/res_scho.pkl'
if os.path.exists(file):
    sub = (pickle.load(open(file,'rb'))['res_var']).toarray() 
    sub = sub.reshape(im_final.shape)
    im_final = np.where(sub!=0,sub,im_final)
    print('替换2完成')
end_time=time.time()
print('总耗时',end_time-start_time,'s')
od = np.load('data_pre/ERA5test2.npy')[:,:36,:720,:]

aaa = od - im_final
percent = abs(aaa/od)
for var in range(5):
    for plevel in range(im_final.shape[1]):
        print('第',var,'变量的',plevel,'层的 RMSE',(((aaa[var,plevel]**2).mean())**0.5))
        print('第',var,'变量的',plevel,'层的 平均绝对误差',abs(aaa[var,plevel]).mean())
        print('第',var,'变量的',plevel,'层的 最大相对误差',(percent[var,plevel]).max())
        print('第',var,'变量的',plevel,'层的 平均相对误差',(percent[var,plevel]).mean())
        # print('第',var,'变量的',plevel,'层的 最大误差值',(aaa2[var,plevel].flatten()[np.argmax(temp2[var,plevel])]))
        # print('第',var,'变量的',plevel,'层的 最大误差值原值',(od[var,plevel].flatten()[np.argmax(temp2[var,plevel])]))

print('draw a Finaldemo')

with torch.no_grad():
    fig, axs = plt.subplots(5, 5, figsize=(8, 4))
    for var in range(5):
        for k in range(5):
            axs[var, k].imshow(im_final[var,k], cmap='viridis')
            # axs[2*var+1, k].imshow(im_global_norm[var,k], cmap='viridis')
    # plt.tight_layout()
    plt.savefig('output_uncompress.png', dpi = 300)

print('计算PSNR')

import math

def cal_psnr():
    quant_output = im_final#[:3,15] 
    # for var in range(quant_output.shape[0]):
    #     quant_output[var] = (quant_output[var]-quant_output[var].min())/(quant_output[var].max()-quant_output[var].min())
    # quant_output = 255*quant_output

    compiled_output = od#[:3,15] #np.fromfile("./output_mc50/dump_ocm_hsk_output_nhwc.raw")
    # for var in range(compiled_output.shape[0]):
    #     compiled_output[var] = (compiled_output[var]-compiled_output[var].min())/(compiled_output[var].max()-compiled_output[var].min())
    # compiled_output = 255*compiled_output

    diff = quant_output - compiled_output
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps
    else:
    	# 65535 = 2^16-1， because original data type is uint16 
        rmse = 20*math.log10(65535.0/rmse)
    print(f"psnr = {rmse}")

cal_psnr()
