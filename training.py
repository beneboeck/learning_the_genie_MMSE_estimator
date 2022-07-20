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
import time

def train_trial1(epochs,dataloader,dataset,model,device,optim,log_file):
    risk = np.zeros(int(np.floor(epochs/30)))
    r=0
    for step in range(epochs):
        if step == 0:
            start = time.time()
        if step == 1:
            stop = time.time()
            print(f'estimated time: {(stop-start)/3600 * epochs} h')
        for C_in, C in dataloader:
            C_in, C = C_in.to(device), C.to(device)
            # angles = math.pi * my_net(h)
            C_learned = model(C_in)
            loss = (torch.abs((C_learned - C)) ** 2).sum(dim=(1, 2)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        if step % 30 == 0:
            with torch.no_grad():
                model.eval()
                C_learned = model(dataset.C_hat[:1000,:,:,:].to(device))
                print(C_learned[0, :5, :5])
                print(dataset.C_hat[0, :5, :5].to(device))
                loss = (torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).sum(dim=(1, 2)).mean()
                risk[r] = np.array(loss.to('cpu'))
                r = r+1
                print(f'total mean loss {(torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).mean():.2f}, step {step}, total loss {loss:.2f}')
                log_file.write(f'total mean loss {(torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).mean():.2f}, step {step}, total loss {loss:.2f}\n')
                model.train()

    return model,risk,log_file

def eval_trial(dataset,model,device,key_file):
    with torch.no_grad():
        model.eval()
        C_learned = model(dataset.C_hat.to(device))
        loss = (torch.abs((C_learned - dataset.C_sim.to(device))) ** 2).sum(dim=(1, 2)).mean()
        key_file.write(f'loss for evaluation set: {loss}\n')