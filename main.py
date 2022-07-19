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

dataset_trail1_train = ds.dataset_trail1(h_train,C_sim_train)


print(h_train.shape)
lr,n_layers,n_conv,n_fully,kernel_size = nas.trail_1_NAS()

my_network = n.trail_1_network(n_conv,n_fully,kernel_size)

print(lr)
print(n_layers)
print(n_conv)
print(n_fully)
print(kernel_size)