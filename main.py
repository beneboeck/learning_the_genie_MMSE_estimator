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
import training as tr
import os
import datetime

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

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
path = '/home/ga42kab/lrz-nashome/learning_the_genie_MMSE_estimator/models/models_trial1'
dir_path = path + '/' + time
os.mkdir (dir_path)

log_file = open(dir_path + '/log_file.txt','w')
key_file = open(dir_path + '/key_file.txt','w')

dataset_trial1_train = ds.dataset_trial1(h_train,C_sim_train)
dataset_trial1_val = ds.dataset_trial1(h_val,C_sim_val)
dataset_trial1_test = ds.dataset_trial1(h_test,C_sim_test)

dataloader_trial1_train = DataLoader(dataset_trial1_train,batch_size=64,shuffle=True)
lr,n_layers,n_conv,n_fully,kernel_size = nas.trail_1_NAS()

epochs = 20
trial = 3
n_coherence = 10
n_antennas = 64

key_file.write('Network Parameters\n')
key_file.write(f'trial: {trial}\n')
key_file.write(f'lr: {lr}\n')
key_file.write(f'n_antennas: {n_antennas}\n')
key_file.write(f'n_layers: {n_layers}\n')
key_file.write(f'n_conv: {n_conv}\n')
key_file.write(f'n_fully: {n_fully}\n')
key_file.write(f'kernel_size: {kernel_size}\n')
network_params = np.array([lr,64,n_conv,n_fully,kernel_size],dtype=np.float32)
np.save(dir_path + '/network_params',network_params)

print('NETWORK PARAMS')
print(lr,n_layers,n_conv,n_fully,kernel_size)

network = n.trial_1_network(64,n_conv,n_fully,kernel_size,device).to(device)
log_file.write('NETWORK:\n')
log_file.write(str(network.net) + '\n')

optim = torch.optim.Adam(lr=lr, params=network.parameters())
network,risk,log_file = tr.train(epochs,trial,n_coherence,dataloader_trial1_train,dataset_trial1_train,network,device,optim,log_file)
save_risk(risk,dir_path)
torch.save(network.state_dict(),dir_path + '/model_dict')

tr.eval_trial(dataset_trial1_val,trial,n_coherence,network,device,key_file)
