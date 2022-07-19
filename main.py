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

path = '/home/ga42kab/lrz-nashome/learning_the_genie_MMSE_estimator/data/'

h_train = np.load(path + 'h_train.npy')
h_test = np.save(path + 'h_test.npy')
h_val = np.save(path + 'h_val.npy')


angles_train = np.save(path + 'angles_train.npy')
angles_test = np.save(path + 'angles_test.npy')
angles_val = np.save(path + 'angles_val.npy')

gains_train = np.save(path + 'gains_train.npy')
gains_test = np.save(path + 'gains_test.npy')
gains_val = np.save(path + 'gains_val.npy')

C_sim_train = np.save(path + 'C_BS_train.npy')
C_sim_test = np.save(path + 'C_BS_test.npy')
C_sim_val = np.save(path + 'C_BS_val.npy')


lr,n_layers,n_conv,n_fully,kernel_size = nas.trail_1_NAS()

print(lr)
print(n_layers)
print(n_conv)
print(n_fully)
print(kernel_size)