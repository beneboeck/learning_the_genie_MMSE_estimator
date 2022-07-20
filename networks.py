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
    def __init__(self,n_antennas,n_conv,n_fully,kernel_size,device):
        super().__init__()
        self.net = []
        in_channels = 2
        for c in range(n_conv):
            self.net.append(nn.Conv2d(int(in_channels),int(2*in_channels),kernel_size,stride=2,padding=int((kernel_size-1)/2)))
            self.net.append(nn.ReLU())
            self.net.append(nn.BatchNorm2d(int(2*in_channels)))
            in_channels = in_channels * 2
        if n_conv > 0:
            in_full = in_channels * 64**2/(2**n_conv)
        else:
            in_full = 64**2 * 2
        self.net.append(nn.Flatten())
        step = round((in_full - 128)/n_fully)
        for n in range(n_fully-1):
            self.net.append(nn.Linear(int(in_full),int(in_full - step)))
            self.net.append(nn.ReLU())
            in_full = round(in_full - step)
        self.net.append(nn.Linear(int(in_full),128))
        self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)

        self.n_antennas = n_antennas
        indices = torch.arange(n_antennas).to(device)
        X, Y = torch.meshgrid(indices, indices)
        self.X = X[None, :, :].to(device)
        self.Y = Y[None, :, :].to(device)
        self.diags = torch.zeros(n_antennas,n_antennas,n_antennas).to(device)
        self.diags[0, :, :] = torch.diag_embed(torch.tensor(1).repeat(1, n_antennas), offset=0)
        for n in range(1,n_antennas):
            self.diags[n,:,:] = torch.diag_embed(torch.tensor(1).repeat(1,n_antennas - n),offset=n) + torch.diag_embed(torch.tensor(1).repeat(1,n_antennas - n),offset=-n)
        self.diags = self.diags[None,:,:,:]

    def compute_learned_cov(self,a_1, a_2):
        A_1 = torch.einsum('ijhl,ij->ihl',self.diags,a_1)
        A_2 = torch.einsum('ijhl,ij->ihl', self.diags, a_2)
        print(A_1[0,:5,:5])
        print(((self.X - self.Y) * A_1)[0,:5,:5])
        return torch.cos(-math.pi * (self.X - self.Y) * A_1) + 1j * torch.sin(-math.pi * (self.X - self.Y) * A_2)

    def forward(self,C):
        a_1,a_2 = self.net(C).chunk(2,dim=1)
        C_learned = self.compute_learned_cov(a_1,a_2)
        return C_learned





