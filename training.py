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

def loss_likelihood(C_hat,C_learned,n_coherence):
    La,U = torch.linalg.eigh(C_learned)
    if (La < 0).sum() != 0:
        print('some eigenvalues were negative!')
        print(torch.min(La[La<0]))
        #La[La < 0] = 1e-5
    La_inv = 1/La
    C_learned_inv = U @ torch.diag_embed(La_inv.cfloat()) @ U.mH
    log_det_C_learned = torch.log(torch.sum(La,dim=1))
    print(log_det_C_learned.size())
    print( torch.einsum('jii->j',C_hat @ C_learned_inv).size())
    loss = -n_coherence * log_det_C_learned - (n_coherence-1) * torch.einsum('jii->j',C_hat @ C_learned_inv)
    return loss

def train(epochs,trial,n_coherence,dataloader,dataset,model,device,optim,log_file):
    risk = []
    for step in range(epochs):
        print(f'new step {step}')
        if step == 0:
            start = time.time()
        if step == 1:
            stop = time.time()
            print(f'estimated time: {(stop-start)/3600 * epochs} h')
        for C_in, C in dataloader:
            C_in, C = C_in.to(device), C.to(device)
            # angles = math.pi * my_net(h)
            C_learned = model(C_in)
            if trial == 1:
                loss = (torch.abs((C_learned - C)) ** 2).sum(dim=(1, 2)).mean()
            if trial == 3:
                loss = loss_likelihood(C_in,C_learned,n_coherence)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if step % 5 == 0:
            with torch.no_grad():
                model.eval()
                C_learned = model(dataset.C_hat[:1000,:,:,:].to(device))
                print(C_learned[0, :3, :3])
                print(dataset.C_sim[0, :3, :3].to(device))
                loss = (torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).sum(dim=(1, 2)).mean()
                risk.append(np.array(loss.to('cpu')))
                print(f'total mean squared error {(torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).mean():.5f}, step {step}, actual loss {loss:.3f}')
                log_file.write(f'total mean squared error {(torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).mean():.5f}, step {step}, actual loss {loss:.3f}\n')
                model.train()

    return model,np.array(risk),log_file



def eval_trial(dataset,trial,n_coherence,model,device,key_file):
    with torch.no_grad():
        model.eval()
        C_learned = model(dataset.C_hat.to(device))
        if trial == 1:
            loss = (torch.abs((C_learned - dataset.C_sim.to(device))) ** 2).sum(dim=(1, 2)).mean()
        if trial == 3:
            loss = loss_likelihood(dataset.C_hat.to(device), C_learned, n_coherence)
        mean_loss = (torch.abs((C_learned - dataset.C_sim.to(device))) ** 2).mean()
        key_file.write(f'loss for evaluation set: {loss}\n')
        key_file.write(f'squared error for evaluation set (elementwise): {mean_loss}\n')