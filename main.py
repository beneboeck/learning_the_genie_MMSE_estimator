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

y_train = np.load(path + 'y_train_100000.npy')
y_test = np.load(path + 'y_test_10000.npy')
y_val = np.load(path + 'y_val_10000.npy')

h_train = np.load(path + 'h_train_100000.npy')
h_test = np.load(path + 'h_test_10000.npy')
h_val = np.load(path + 'h_val_10000.npy')

angles_train = np.load(path + 'angles_train_100000.npy')
angles_test = np.load(path + 'angles_test_10000.npy')
angles_val = np.load(path + 'angles_val_10000.npy')

gains_train = np.load(path + 'gains_train_100000.npy')
gains_test = np.load(path + 'gains_test_10000.npy')
gains_val = np.load(path + 'gains_val_10000.npy')

C_sim_train = np.load(path + 'C_BS_train_100000.npy')
C_sim_test = np.load(path + 'C_BS_test_10000.npy')
C_sim_val = np.load(path + 'C_BS_val_10000.npy')

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
path = '/home/ga42kab/lrz-nashome/learning_the_genie_MMSE_estimator/models/models_trial4'
dir_path = path + '/' + time + '_100000'
os.mkdir (dir_path)

log_file = open(dir_path + '/log_file_100000.txt','w')
key_file = open(dir_path + '/key_file_100000.txt','w')

dataset_trial1_train = ds.dataset_trial1(h_train,C_sim_train,y_train)
dataset_trial1_val = ds.dataset_trial1(h_val,C_sim_val,y_val)
dataset_trial1_test = ds.dataset_trial1(h_test,C_sim_test,y_test)

dataloader_trial1_train = DataLoader(dataset_trial1_train,batch_size=64,shuffle=True)
lr,n_layers,n_conv,n_fully,kernel_size,dropout_bool,drop_prob = nas.trail_1_NAS()

epochs = 100
trial = 4
n_coherence = 10
n_antennas = 64
SNR_db = 5
SNR_eff = 10**(SNR_db/10)
sig_n = math.sqrt(1/(n_antennas * SNR_eff))

key_file.write('Network Parameters\n')
key_file.write(f'trial: {trial}\n')
key_file.write(f'lr: {lr}\n')
key_file.write(f'n_antennas: {n_antennas}\n')
key_file.write(f'n_layers: {n_layers}\n')
key_file.write(f'n_conv: {n_conv}\n')
key_file.write(f'n_fully: {n_fully}\n')
key_file.write(f'kernel_size: {kernel_size}\n')
key_file.write(f'dropout_bool: {dropout_bool}\n')
key_file.write(f'drop_prob: {drop_prob}\n')
network_params = np.array([lr,64,n_conv,n_fully,kernel_size,dropout_bool,drop_prob],dtype=np.float32)
np.save(dir_path + '/network_params',network_params)

print('NETWORK PARAMS')
print(lr,n_layers,n_conv,n_fully,kernel_size,dropout_bool,drop_prob)

network = n.trial_1_network(n_antennas,n_conv,n_fully,kernel_size,dropout_bool,drop_prob,device).to(device)
log_file.write('NETWORK:\n')
log_file.write(str(network.net) + '\n')

optim = torch.optim.Adam(lr=lr, params=network.parameters())
network,risk,eval_risk,log_file = tr.train(epochs,trial,n_coherence,sig_n,dataloader_trial1_train,dataset_trial1_train,dataset_trial1_val,network,device,optim,log_file)
save_risk(risk,dir_path,'Risk')
save_risk(eval_risk,dir_path,'Eval Risk')
torch.save(network.state_dict(),dir_path + '/model_dict')

tr.eval_trial(dataset_trial1_test,trial,n_coherence,sig_n,network,device,key_file)
