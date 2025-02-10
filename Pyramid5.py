import torch
from bisect import bisect_right
from torch import nn
import numpy as np
from Siren_simple import Siren as Siren_main
import cmaps
import os
import random
import torch.optim as optim
from tqdm import tqdm
import logging
import time
from scipy import sparse
from utils import Siren_adaptive
import pickle
import yaml
import sys
import copy
from img_frequ import fre_pixel

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

# Loading configuration file 加载配置文件
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

hidden_size_top = config['hidden_size_top'] 
hidden_size_topres = config['hidden_size_topres'] #256
hidden_size_L4 = config['hidden_size_L4'] #256 #384
hidden_size_L2 = config['hidden_size_L2'] #128
layer_top = config['layer_top'] #2
layer_topres = config['layer_topres'] #1
layer_L4 = config['layer_L4'] #1
layer_L2 = config['layer_L2'] #1
fw0_top = config['fw0_top'] #10.
hw0_top = config['hw0_top'] #14.
fw0_topres = config['fw0_topres'] #22.
hw0_topres = config['hw0_topres']#16.
fw0_L4 = config['fw0_L4']  #15. #150.
hw0_L4 = config['hw0_L4']  #15. #150.
fw0_L2 = config['fw0_L2']  #30.
hw0_L2 = config['hw0_L2']  #30.

# Create 3-D axis data
def get_data(nscale_x,nscale_y,nscale_z,mask):
    xx,yy,zz = torch.meshgrid(torch.linspace(-1, 1, nscale_x),torch.linspace(-1, 1, nscale_y),torch.linspace(-1, 1, nscale_z),indexing='ij')
    # CooRds = torch.flatten(torch.cat((xx.unsqueeze(0),yy.unsqueeze(0),zz.unsqueeze(0)),0),-3,-1).permute(1,0) #把前三维合并
    CooRds = torch.ones(3,xx.shape[0]*xx.shape[1]*xx.shape[2])
    coords = torch.cat((xx.unsqueeze(0),yy.unsqueeze(0),zz.unsqueeze(0)),0)
    for idx in range(3):
        CooRds[idx,:] = torch.masked_select(coords[idx],mask.cpu())
    CooRds = CooRds.permute(1,0)
    return CooRds
# Create non-mask data
def get_data_nomask(nscale_x,nscale_y,nscale_z):
    xx,yy,zz = torch.meshgrid(torch.linspace(-1, 1, nscale_x),torch.linspace(-1, 1, nscale_y),torch.linspace(-1, 1, nscale_z),indexing='ij')
    CooRds = torch.flatten(torch.cat((xx.unsqueeze(0),yy.unsqueeze(0),zz.unsqueeze(0)),0),-3,-1).permute(1,0) #把前三维合并
    return CooRds
# Create mask data
def get_mask(input,mask):
    output = torch.ones(input.shape[0],input.shape[1]*input.shape[2]*input.shape[3])
    for idx in range(input.shape[0]):
        output[idx,:] = torch.masked_select(input[idx],mask.cpu()).permute(1,0)
    return output

def mae(a,b):
    return abs((a-b)**2).mean()
# Divide data into sub-chunks
def sub_chunk_check(constant,data,target):
    ncale_x_level4 = int(data.shape[1]/constant)
    ncale_y_level4 = int(data.shape[2]/constant)
    ncale_z_level4 = int(data.shape[3]/constant)
    chunk_num = constant**3
    for chunk_idx in tqdm(range(chunk_num)):
        x,y,z = np.unravel_index(chunk_idx,(constant,constant,constant)) #反索引一下三维矩阵
        ck1 = data[...,int(x*ncale_x_level4):int((x+1)*ncale_x_level4),int(y*ncale_y_level4):int((y+1)*ncale_y_level4),int(z*ncale_z_level4):int((z+1)*ncale_z_level4)]
        ck2 = target[...,int(x*ncale_x_level4):int((x+1)*ncale_x_level4),int(y*ncale_y_level4):int((y+1)*ncale_y_level4),int(z*ncale_z_level4):int((z+1)*ncale_z_level4)]
        resduil = abs(ck1-ck2).mean()
        if chunk_idx == 0:
            max_res = resduil
            max_res_index = 0
            max_res_index = chunk_idx
            index_x = x
            index_y = y
            index_z = z
            chunk = ck2

        if resduil>max_res:
            max_res = resduil
            max_res_index = chunk_idx
            index_x = x
            index_y = y
            index_z = z
            chunk = ck2
    return index_x,index_y,index_z,chunk,max_res_index
# INR with Laplacian pyramid
def pyramid(savingpath,im,Nchunk,acc_top,acc_L4,acc_L2,optimsig,device,parallel,succeed,prev_path):
    dir_path = savingpath
    im_top = im.to(device)
    origin = im_top.clone()
    var_num = im_top.shape[0]
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    criterion2 = nn.MSELoss()
    criterion = nn.L1Loss()
    criterion3 = nn.SmoothL1Loss()
 # define the train process
    def train_top(data,Label1,Model,optimizer,sub_item):
        tot_loss = 0
        # data = data
        target = torch.flatten(Label1,-3,-1).permute(1,0)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,coords = Model(data) 
        if sub_item!=None:
            sub_item = sub_item.to(device)
            output = output.view((ncale_x,ncale_y,ncale_z,var_num)).permute(3,0,1,2)
            output = torch.nn.functional.interpolate(output[None,...], scale_factor=4, mode='trilinear').squeeze()
            output = torch.where(sub_item!=0,sub_item,output)
            output = torch.flatten(output,-3,-1).permute(1,0)
        SL1loss = (output-target).max() #criterion2(output, target)
        minloss = (output-target).min()
        # mse = criterion(output, target)
        loss = criterion(output, target) #(((output-target)**2).mean())**0.5  #criterion(output, target) #abs(((output-target)/target).mean()) #criterion2(output,target)

        loss.backward()#calcluate the gradient by backpropagation
        optimizer.step()#Gradient descent
        # tot_loss += loss.item()
        return loss.item(), output, SL1loss, minloss

    ###################################### top of pyramid ##################################
    lR = 40 #35
    print('lR',lR)
    lr = lR*(1e-5)
    to_continue = True
    first_time = True
    hidden_size_top1 = hidden_size_top

    im_topB = np.array(im_top.cpu())
    fre = np.zeros(im_top.shape)
    print('Making harmonic decomposition')
    # Harmonic decomposition part
    for index1 in tqdm(range(im_topB.shape[-2])):
        for index2 in range(im_topB.shape[-1]):
            fre[...,index1,index2] = fre_pixel(index1,index2,im_topB)
    #define decomposition threshold
    high_fre = 3*fw0_top*(6/5)**0.5
    low_fre = fw0_top*(6/5)**0.5
    # Harmonic decomposition
    sub = np.where(fre>high_fre, im_topB, 0) #high
    low_part = np.where(fre<low_fre, im_topB, 0) #low
    mid = fre-sub-low_part #mid
    ################################################
    for var in range(im_top.shape[0]):
        for level in range(im_top.shape[1]):
            im_top[var,level]=torch.tensor(np.where(fre[var,level]<0.5, im_topB[var,level], im_topB[var,level].mean()))
    suB = sub.reshape(sub.shape[0]*sub.shape[1]*sub.shape[2],sub.shape[3])
    allmatrix_sp=sparse.csr_matrix(suB)
    res_path = savingpath + '/high_fre'+str(Nchunk)+'.pkl'
    dic = {'res_var':allmatrix_sp}
    file = open(res_path, 'wb')
    pickle.dump(dic, file)

    nscale_top = 3
    ncale_x = int(im_top.shape[1]/nscale_top) #8
    ncale_y = int(im_top.shape[2]/nscale_top)
    ncale_z = int(im_top.shape[3]/nscale_top)
    im_top_LABEL = torch.nn.functional.interpolate(im_top[None,...], scale_factor=1/nscale_top, mode='trilinear').squeeze()
    # im_top_LABEL = im_top
    im_list = []
    # for plevel in range(ncale_x):
        # im_list.append(torch.nn.functional.interpolate(im_top[:,None,plevel,...], scale_factor=1/nscale_top, mode='bilinear'))

    # im_top_LABEL = torch.cat(im_list,1) #im_top1.squeeze()

    decay = 0.0005
    N_para = 0
    print('现在的channel是',hidden_size_top1)
    # model_top = Siren_main( in_features=3,hidden_features=hidden_size_top,hidden_layers=layer_top,out_features=var_num,outermost_linear=True, first_omega_0=fw0_top, hidden_omega_0=hw0_top) 
    model_top = Siren_adaptive(3,var_num,hidden_size_top1,layer_top,fw0_top,hw0_top)
    if parallel:
        model_top = nn.DataParallel(model_top)
    # device = torch.device('cuda')
    model_top = model_top.to(device)
    if optimsig == "sgd":
        optimizer_top = optim.SGD(lr=lr, params=model_top.parameters())
    elif optimsig == "adam":
        optimizer_top = optim.Adam(lr=lr, params=model_top.parameters())
    max_epoch = 20000 #4000
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=int(0.5*max_epoch), eta_min=lr/10)
    top_path = savingpath+'/pooling_para_'+str(Nchunk)+'.pth'
    if succeed and os.path.exists(prev_path):
        prev_path1 = prev_path+'/pooling_para_'+str(Nchunk)+'.pth'
        if os.path.exists(prev_path1):
            model_top.load_state_dict(torch.load(prev_path1)) #读取上一个chunk的参数
    best_loss = 10
    loss_list = []
    input = get_data_nomask(ncale_x,ncale_y,ncale_z)
    input = input.to(device)
    for epoch in tqdm(range(max_epoch)):
        model_top.train()
        loss,out,maxloss,minloss = train_top(input,im_top_LABEL,model_top,optimizer_top,None)
        scheduler.step() 
        loss_list.append(loss)
        # print('mae=',loss)
        if loss < best_loss:
            # print('mae',loss)
            best_loss = loss
            best_model = copy.deepcopy(model_top.state_dict())
            out_best = out.clone().detach()
        if epoch == int(0.5*max_epoch):
            epoch_ = epoch
            optimizer_top = optim.AdamW(lr=lr/4, params=model_top.parameters())
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=500, eta_min=lr/8)
        if epoch == int(0.9*max_epoch):
            optimizer_top = optim.Adam(lr=lr/8, params=model_top.parameters())
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_top, T_max=3000, eta_min=lr/12)
        if loss < acc_top: # and maxloss < 0.1 and minloss > -0.1: #here
            model_top.eval()
            img_top = out.detach().view((ncale_x,ncale_y,ncale_z,var_num)).permute(3,0,1,2)
            torch.save(model_top.state_dict(), top_path)
            # del model_top
            break #金字塔顶达到top结束
        elif epoch == max_epoch-1:
            torch.save(best_model, top_path)
            img_top=out_best.detach().view((ncale_x,ncale_y,ncale_z,var_num)).permute(3,0,1,2)
            # del model_top
    print ('stage1 mae =', best_loss)
    torch.cuda.empty_cache()
    del model_top

    ################### Top of pyramid #########################

    im_L4_list = []
    # make a interpolation
    t11 = torch.nn.functional.interpolate(img_top[None,...], scale_factor=nscale_top, mode='trilinear').squeeze()
    # for plevel in range(ncale_x):
    #         im_L4_list.append(torch.nn.functional.interpolate(img_top[:,None,plevel,...], scale_factor=nscale_top, mode='bilinear'))
    # t11 = torch.cat(im_L4_list,1)
    img_L4_np = np.array(t11.cpu())
    print(img_L4_np.shape)
    img_L4_np = np.where(sub!=0,sub,img_L4_np)
    im_toP = np.array(im_top.cpu())
    MMAE = abs(np.array(im_top.cpu())-img_L4_np)
    top_sub = np.where(MMAE>0.5, im_toP, 0)
    img_L4_np = np.where(top_sub!=0, top_sub, img_L4_np)
    top_suB = top_sub.reshape(top_sub.shape[0]*top_sub.shape[1]*top_sub.shape[2],top_sub.shape[3])
    allmatrix_sp=sparse.csr_matrix(top_suB)
    res_path = savingpath + '/res_scho_top'+str(Nchunk)+'.pkl'
    dic = {'res_var':allmatrix_sp}
    file = open(res_path, 'wb')
    pickle.dump(dic, file)#转换成稀疏矩阵存储，会小很多
    img_L4 = torch.tensor(img_L4_np).to(device)
    
    print('top is finished')

    ###################################### pyramid level 1 ##############################    
    nscales = 1
    ncale_x_level4 = int(im_top.shape[1]/nscales)
    ncale_y_level4 = int(im_top.shape[2]/nscales)
    ncale_z_level4 = int(im_top.shape[3]/nscales)

    # avg3d_4 = torch.nn.AdaptiveAvgPool3d((ncale_x_level4,ncale_y_level4,ncale_z_level4))
    # polling the image
    
    # im_L4 = avg3d_4(im_top.unsqueeze(0)).squeeze().to(device)
    n_para = 0
    for chunk_idx in tqdm(range(nscale_top**3)):
        x,y,z = np.unravel_index(chunk_idx,(nscale_top,nscale_top,nscale_top)) #re-index the coordinex of data
        ck1 = im_top[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)]
        ck2 = img_L4[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)]
        # sub_chunk = im_top_LABEL[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)]
        resduil = ck1-ck2 #原先是ck1-ck2
        mmse = abs(resduil).mean() 
        #find the chunk of the certain index
        # mmse = (resduil**2).mean()
        if mmse > acc_L4:
        #Define a single model for every chunk
            model_name = savingpath+'/L4_' + str(chunk_idx)+'_' + str(Nchunk)+'.pth'
            model_level_4 = Siren_main(in_features=3,hidden_features=hidden_size_L4,hidden_layers=layer_L4,out_features=5,outermost_linear=True, hidden_omega_0=60.)
            max_epoch = 2000
            if parallel:
                model_level_4 = nn.DataParallel(model_level_4)
            model_level_4 = model_level_4.to(device)
            if optimsig == "sgd":
                optimizer1 = optim.SGD(lr=lr, params=model_level_4.parameters())
            else:
                optimizer1 = optim.Adam(lr=5e-4, params=model_level_4.parameters())
            scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=int(max_epoch*0.5), eta_min=5e-6)
            model_level_4.train()
            n_para += get_parameter_number(model_level_4)["Total"]
            input = get_data_nomask(ncale_x_level4//nscale_top,ncale_y_level4//nscale_top,ncale_z_level4//nscale_top)
            input = input.to(device)
            for epoch in range(max_epoch):
                loss, output_L4,maxloss,minloss = train_top(input,resduil,model_level_4,optimizer1,None)
                # print('stage 2 mae=',loss)
                scheduler1.step()
                if loss < acc_L4: #here
                    model_level_4.eval()
                    out_L2 = output_L4.detach().view((ncale_x,ncale_y,ncale_z,5)).permute(3,0,1,2)
                    torch.save(model_level_4.state_dict(), model_name)
                    img_L4[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)] += out_L2
                    torch.cuda.empty_cache()
                    del model_level_4
                    break
                if epoch == max_epoch-1: #here
                    model_level_4.eval()
                    out_L2 = output_L4.detach().view((ncale_x,ncale_y,ncale_z,5)).permute(3,0,1,2)
                    torch.save(model_level_4.state_dict(), model_name)
                    img_L4[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)] += out_L2
                    torch.cuda.empty_cache()
                    del model_level_4
                    break
            Error_L4 = abs((img_L4[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)]-ck1))
            sub_L4 = torch.where(Error_L4>0.1,ck1,0)
            img_L4[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)] = torch.where(sub_L4!=0,sub_L4,img_L4[...,int(x/nscale_top*ncale_x_level4):int((x+1)/nscale_top*ncale_x_level4),int(y/nscale_top*ncale_y_level4):int((y+1)/nscale_top*ncale_y_level4),int(z/nscale_top*ncale_z_level4):int((z+1)/nscale_top*ncale_z_level4)])
            sub_L4 = np.array(sub_L4.cpu())
            sub_L4B = sub_L4.reshape(sub_L4.shape[0]*sub_L4.shape[1]*sub_L4.shape[2],sub_L4.shape[3])
            allmatrix_sp=sparse.csr_matrix(sub_L4B)
            res_path = savingpath + '/res_scho_L4_'+str(chunk_idx)+'_'+str(Nchunk)+'.pkl'
            dic = {'res_var':allmatrix_sp}
            file = open(res_path, 'wb')
            pickle.dump(dic, file) #Translate to sparse matrix

    # img_L2 = torch.nn.functional.interpolate(img_L4[None,...], scale_factor=2, mode='trilinear')
    # img_L2 = img_L2.squeeze()
    print('L4 is finished')
    im_out = img_L4
    # print("第二层模型总大小为：{:.3f}Million".format(n_para/1000000))
    # N_para+=n_para/1000000
    im_out = np.array(im_out.cpu())
    im_out = np.where(top_sub!=0, top_sub, im_out)
    im_out = np.where(sub!=0,sub,im_out)
    print('overall_MAE=',abs((im_out-np.array(origin.cpu()))).mean())

    return torch.tensor(im_out)

# uncompress process
def pyramid_uncompress(pthfile,plevel,lat,lon,Nchunk,device,parallel):

    var_num = 5

    nscale_top = 1
    ncale_x = int(plevel/nscale_top) #8
    ncale_y = int(lat/nscale_top)
    ncale_z = int(lon/nscale_top)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    ###################################### top of pyramid ##################################

    # model_top = Siren_main( in_features=3,hidden_features=hidden_size_top,hidden_layers=layer_top,out_features=var_num,outermost_linear=True, first_omega_0=fw0_top, hidden_omega_0=hw0_top) 
    model_top = Siren_adaptive(3,var_num,hidden_size_top,layer_top,fw0_top,hw0_top)
    if parallel:
        model_top = nn.DataParallel(model_top)
    model_top = model_top.to(device)
    top_path = pthfile+'/pooling_para_'+str(Nchunk)+'.pth'

    if os.path.exists(top_path):
        model_top.load_state_dict(torch.load(top_path)) #读取上一个文件的参数
        model_top.train()
        coords = get_data(ncale_x,ncale_y,ncale_z).to(device)
        out_best,coords = model_top(coords) 
        # loss,out,maxloss,minloss = train_top(get_data(ncale_x,ncale_y,ncale_z),im_top,model_top,optimizer_top)
        img_top=out_best.detach().view((ncale_x,ncale_y,ncale_z,var_num)).permute(3,0,1,2)
        del model_top
        torch.cuda.empty_cache()
############substitution1##############
    resfile = pthfile+'/res_top'+str(Nchunk)+'.pkl'
    if os.path.exists(resfile):
        M = pickle.load(open(resfile,'rb'))
        M = M['res_var'].toarray()
        M = torch.tensor(M.reshape(var_num,ncale_x,ncale_y,ncale_z))
        M = M.to(device)
        img_top += M
#############Residual error 1############
    top_path2 = pthfile+'/pooling_para_res_'+str(Nchunk)+'.pth'
    if os.path.exists(top_path2):
        model_top2 = Siren_adaptive(5,var_num,hidden_size_topres,layer_topres,fw0_topres,hw0_topres)
        model_top2.eval()
        out_best,coords = model_top2(torch.flatten(img_top,-3,-1).permute(1,0)) 
        # loss,out,maxloss,minloss = train_top(get_data(ncale_x,ncale_y,ncale_z),im_top,model_top,optimizer_top)
        img_top = out_best.detach().view((ncale_x,ncale_y,ncale_z,var_num)).permute(3,0,1,2)
        del model_top2
        torch.cuda.empty_cache()
############substitution2##############
    resfile = pthfile+'/res_topres'+str(Nchunk)+'.pkl'
    if os.path.exists(resfile):
        M = pickle.load(open(resfile,'rb'))
        M = M['res_var'].toarray()
        M = torch.tensor(M.reshape(var_num,ncale_x,ncale_y,ncale_z))
        M = M.to(device)
        img_top+=M
###########Residual error 2##############
    top_path3 = pthfile+'/pooling_para_res2_'+str(Nchunk)+'.pth'
    if os.path.exists(top_path3):
        model_top3 = Siren_adaptive(3,var_num,hidden_size_topres,layer_topres,fw0_topres,hw0_topres)
        model_top3.eval()
        out_best,coords = model_top3(get_data(ncale_x,ncale_y,ncale_z)) 
        # loss,out,maxloss,minloss = train_top(get_data(ncale_x,ncale_y,ncale_z),im_top,model_top,optimizer_top)
        img_top += out_best.detach().view((ncale_x,ncale_y,ncale_z,var_num)).permute(3,0,1,2)
        del model_top2
        torch.cuda.empty_cache()

    return img_top
