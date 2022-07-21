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

def compute_sim_cov(t_BS,n_antennas):
    t_BS = torch.tensor(t_BS)
    n_sample = t_BS.size(0)
    C_sim = torch.zeros((n_sample,n_antennas,n_antennas),dtype=torch.cfloat)
    C_sim[:,0,:] = t_BS
    for i in range(1,n_antennas):
        C_sim[:,i,:] = torch.roll(t_BS,i,dims=1)
    C_sim = torch.triu(C_sim) + torch.tril(torch.conj(torch.transpose(C_sim,dim0 = 2,dim1 = 1))) - torch.eye(n_antennas)
    return np.array(C_sim)

n_coherence = 10
N_train = 200000
N_val = 20000
N_test = 20000
n_paths = 3
n_antennas = 64

path = '/home/ga42kab/lrz-nashome/learning_the_genie_MMSE_estimator/data/'
log_file = open(path + 'log_file_data_description_200000.txt','w')
log_file.write(f'n_coherence: {n_coherence}\n')
log_file.write(f'N_train: {N_train}\n')
log_file.write(f'N_val: {N_val}\n')
log_file.write(f'N_test: {N_test}\n')
log_file.write(f'n_paths: {n_paths}\n')
log_file.write(f'n_antennas: {n_antennas}\n')
log_file.close()

channel_generator = cg.SCMMulti(n_path = n_paths)

h_train, t_BS_train, t_MS_train,gains_train,angles_train = channel_generator.generate_channel(n_batches=N_train,n_coherence=n_coherence,n_antennas_BS=n_antennas,n_antennas_MS=1)

h_val, t_BS_val, t_MS_val,gains_val,angles_val = channel_generator.generate_channel(n_batches=N_val,n_coherence=n_coherence,n_antennas_BS=n_antennas,n_antennas_MS=1)

h_test, t_BS_test, t_MS_train,gains_test,angles_test = channel_generator.generate_channel(n_batches=N_test,n_coherence=n_coherence,n_antennas_BS=n_antennas,n_antennas_MS=1)

C_sim_train = compute_sim_cov(t_BS_train,n_antennas)
C_sim_val = compute_sim_cov(t_BS_val,n_antennas)
C_sim_test = compute_sim_cov(t_BS_test,n_antennas)

print(C_sim_train.shape)
print(h_train.shape)
print(gains_train.shape)


np.save(path + 'h_train_200000',h_train)
np.save(path + 'h_test_20000',h_test)
np.save(path + 'h_val_20000',h_val)


np.save(path + 'angles_train_200000',angles_train)
np.save(path + 'angles_test_20000',angles_test)
np.save(path + 'angles_val_20000',angles_val)

np.save(path + 'gains_train_200000',gains_train)
np.save(path + 'gains_test_20000',gains_test)
np.save(path + 'gains_val_20000',gains_val)

np.save(path + 'C_BS_train_200000',C_sim_train)
np.save(path + 'C_BS_test_20000',C_sim_test)
np.save(path + 'C_BS_val_20000',C_sim_val)
