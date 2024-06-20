import torch
from bisect import bisect_right
from torch import nn
import numpy as np
from Siren_simple import Siren as Siren_main
from Siren_kan import Siren_KAN
import cmaps
import os
import random
import torch.optim as optim
from tqdm import tqdm
import logging
import time
from scipy import sparse

hidden_size_top = 156
hidden_size_topres = 256
hidden_size_L4 = 128 #384
hidden_size_L2 = 128
layer_top = 2
layer_topres = 1
layer_L4 = 1
layer_L2 = 1
fw0_top = 10.
hw0_top = 14.
fw0_topres = 15.
hw0_topres = 15.
fw0_L4 = 28. #150.
hw0_L4 = 28. #150.
fw0_L2 = 30.
hw0_L2 = 30.

class Deconv(nn.Module):
    def __init__(self, in_features, out_features,types):
        super().__init__()
        self.type=types
        self.deconv = torch.nn.ConvTranspose3d(in_features,out_features,4,2,1)
        self.deconv2 = torch.nn.ConvTranspose3d(2*in_features,out_features,1,1)
    def forward(self, x1, x2):
        if self.type==4:
            out = self.deconv(x1)
        elif self.type==2:
            data = self.deconv(x1)
            data = torch.cat((data,x2),0)
            out = self.deconv2(data)
        return out

class Siren_adaptive(nn.Module):
    def __init__(self, in_features, out_features, h_size, h_layer, fw0, hw0):
        super().__init__()
        self.siren = Siren_main( in_features=in_features,hidden_features=h_size,hidden_layers=h_layer,out_features=out_features,outermost_linear=True, first_omega_0=fw0, hidden_omega_0=hw0) 
        self.linear = torch.nn.Linear(out_features, out_features, bias=True)
    def forward(self, x):
        x,_coords = self.siren(x)
        # x = self.linear(x)
        return x,_coords
    
class Siren_Kan(nn.Module):
    def __init__(self, in_features, out_features, h_size, h_layer, fw0, hw0):
        super().__init__()
        self.siren = Siren_KAN( in_features=in_features,hidden_features=h_size,hidden_layers=h_layer,out_features=out_features,outermost_linear=True, first_omega_0=fw0, hidden_omega_0=hw0) 
        self.linear = torch.nn.Linear(out_features, out_features, bias=True)
    def forward(self, x):
        x,_coords = self.siren(x)
        # x = self.linear(x)
        return x,_coords

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1/3,
        warmup_iters=100,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
 
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
 
    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]