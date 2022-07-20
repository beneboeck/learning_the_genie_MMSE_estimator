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

def loss_likelihood(C_hat,C_learned,n_coherence,device):
    Q,R = torch.linalg.qr(C_learned)
    R_inv = torch.linalg.solve_triangular(R,torch.eye(64).to(device),upper=True)
    Q_inv = Q.mH
    C_learned_inv = R_inv @ Q_inv
    log_det_C_learned = torch.log(torch.abs(torch.prod(torch.diagonal(R,dim1=1,dim2=2),dim=1)))
    C_hat = torch.complex(C_hat[:,0,:,:],C_hat[:,1,:,:])
    print('test')
    print(torch.diagonal(R,dim1=1,dim2=2).size())
    print((C_learned @ C_learned_inv)[0,:,:])
    print(C_hat.size())
    print(C_learned_inv.size())
    print(log_det_C_learned.size())
    print(torch.log(torch.abs(torch.det(C_learned[0,:,:]))))
    print(log_det_C_learned[0])
    loss = - (-n_coherence * log_det_C_learned - (n_coherence-1) * torch.einsum('jii->j',C_hat @ C_learned_inv)).mean()
    return loss

def train(epochs,trial,n_coherence,dataloader,dataset,dataset_val,model,device,optim,log_file):
    risk = []
    eval_risk_mod = torch.zeros(6)
    eval_risk = []
    e = 0
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
                loss = loss_likelihood(C_in,C_learned,n_coherence,device)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if step % 5 == 0:
            with torch.no_grad():
                model.eval()
                C_learned = model(dataset.C_hat[:1000,:,:,:].to(device))
                print(C_learned[0, :3, :3])
                print(dataset.C_sim[0, :3, :3].to(device))
                if trial == 1:
                    loss = (torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).sum(dim=(1, 2)).mean()
                else:
                    loss = loss_likelihood(dataset.C_hat[:1000,:,:,:].to(device), C_learned, n_coherence, device)
                risk.append(np.array(loss.to('cpu')))
                print(f'total mean squared error {(torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).mean():.5f}, step {step}, actual loss {loss:.3f}')
                log_file.write(f'total mean squared error {(torch.abs((C_learned - dataset.C_sim[:1000,:,:].to(device))) ** 2).mean():.5f}, step {step}, actual loss {loss:.3f}\n')
                La = torch.linalg.eigvalsh(C_learned)
                if (La < 0).sum() != 0:
                    print('some eigenvalues were negative!')
                    print(torch.min(La[La < 0]))
                    log_file.write('some eigenvalues were negative!')
                    log_file.write(str(torch.min(La[La < 0])))

                # early stopping criterion
                C_learned = model(dataset_val.C_hat.to(device))
                if trial == 1:
                    loss = (torch.abs((C_learned - dataset_val.C_sim.to(device))) ** 2).sum(dim=(1, 2)).mean()
                else:
                    loss = loss_likelihood(dataset_val.C_hat.to(device), C_learned, n_coherence, device)
                x = torch.arange(6).to(device)
                eval_risk.append(loss)
                eval_risk_mod[e%6] = loss
                e = e+1
                if step > 100:
                    slope = (x * loss).sum()/((x * x).sum())
                    print('slope')
                    print(slope)
                    log_file.write(f'evaluation loss after step {step}: {slope:.4f}\n')
                model.train()

        if slope > 0:
            log_file.write('BREAKING CONDITION, slope negativ\n')
            log_file.write(f'number epochs: {step}')
            break

    return model,np.array(risk),np.array(eval_risk),log_file



def eval_trial(dataset,trial,n_coherence,model,device,key_file):
    with torch.no_grad():
        model.eval()
        C_learned = model(dataset.C_hat.to(device))
        if trial == 1:
            loss = (torch.abs((C_learned - dataset.C_sim.to(device))) ** 2).sum(dim=(1, 2)).mean()
        if trial == 3:
            loss = loss_likelihood(dataset.C_hat.to(device), C_learned, n_coherence,device)
        mean_loss = (torch.abs((C_learned - dataset.C_sim.to(device))) ** 2).mean()
        key_file.write(f'loss for evaluation set: {loss}\n')
        key_file.write(f'squared error for evaluation set (elementwise): {mean_loss}\n')