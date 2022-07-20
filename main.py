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
import datasets as ds
import networks as n

path = '/home/ga42kab/lrz-nashome/learning_the_genie_MMSE_estimator/data/'

h_train = np.load(path + 'h_train.npy')
h_test = np.load(path + 'h_test.npy')
h_val = np.load(path + 'h_val.npy')


angles_train = np.load(path + 'angles_train.npy')
angles_test = np.load(path + 'angles_test.npy')
angles_val = np.load(path + 'angles_val.npy')

gains_train = np.load(path + 'gains_train.npy')
gains_test = np.load(path + 'gains_test.npy')
gains_val = np.load(path + 'gains_val.npy')

C_sim_train = np.load(path + 'C_BS_train.npy')
C_sim_test = np.load(path + 'C_BS_test.npy')
C_sim_val = np.load(path + 'C_BS_val.npy')

dataset_trial1_train = ds.dataset_trial1(h_train,C_sim_train)
dataset_trial1_val = ds.dataset_trial1(h_val,C_sim_val)
dataset_trial1_test = ds.dataset_trial1(h_test,C_sim_test)

dataloader_trial1_train = DataLoader(dataset_trial1_train,batch_size=64,shuffle=True)
device = 'cuda:0'

print(h_train.shape)
lr,n_layers,n_conv,n_fully,kernel_size = nas.trail_1_NAS()

my_network = n.trail_1_network(64,n_conv,n_fully,kernel_size,device)
a = iter(dataloader_trial1_train)
b,c = a.next()
b = b.to(device)
print(b.size())
print(b[0,0,:5,:5])
output = my_network(b)
print(output[0,:5,:5])

print(lr)
print(n_layers)
print(n_conv)
print(n_fully)
print(kernel_size)