import numpy as np
import SCMMulti_MIMO as cg
import h5py
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
import scipy
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import network_architecture_search as nas


class trail_1_network(nn.Module):
    def __init__(self,n_conv,n_fully,kernel_size):
        super().__init__()
        self.net = []
        in_channels = 2
        for c in range(n_conv):
            self.net.append(nn.Conv2d(in_channels,2*in_channels,kernel_size,stride=2,padding=(kernel_size-1)/2))
            self.net.append(nn.ReLU())
            self.net.append(nn.BatchNorm2d(2*in_channels))
            in_channels = in_channels * 2

        in_full = in_channels * 64**2*2/(2**n_conv)
        step = round((in_full - 128)/n_fully)
        for n in range(n_fully-1):
            self.net.append(nn.Linear(int(in_full),int(in_full - step)))
            self.net.append(nn.ReLU())
            in_full = round(in_full - step)
            print(in_full)
        self.net.append(nn.Linear(int(in_full),128))
        self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)

    def forward(self,C):
        return self.net(C)





