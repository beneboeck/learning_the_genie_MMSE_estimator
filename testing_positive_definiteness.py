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


def compute_learned_cov(theta_1, theta_2):
    A_1 = torch.einsum('ijhl,ij->ihl', diags, theta_1)
    A_2 = torch.einsum('ijhl,ij->ihl', diags, theta_2)
    return torch.cos(-math.pi * (X - Y) * torch.sin(A_1)) + 1j * torch.sin(-math.pi * (X - Y) * torch.sin(A_2))

def checking_pd(C,n_antennas):
    L,U = torch.linalg.eigh(C)
    #print('negative stuff')
    #print(torch.max(torch.abs(torch.imag(L))))
    #print(L[].size())
    print(L.size())
    print((L<0).size())
    print(L<0)
    print((L < 0) * L)
    neg_eig = torch.sum((L<0)*L,dim=1)
    print(neg_eig.size())
    print(neg_eig)
    print(U.size())
    unitary_test = U.mH @ U
    distance = torch.linalg.norm(unitary_test-torch.eye(n_antennas)[None,:,:],dim=(1,2))
    print('distance')
    print(distance.size())

    sol = torch.zeros(L.size(0))
    print(neg_eig >= 0)
    print(distance  < 1e-5)
    sol[(neg_eig >= 0) & (distance < 1e-5)] = 1

    return sol

n_antennas = 2
indices = torch.arange(n_antennas)
X, Y = torch.meshgrid(indices, indices)
X = X[None, :, :]
Y = Y[None, :, :]
diags = torch.zeros(n_antennas,n_antennas,n_antennas)
diags[0, :, :] = torch.diag_embed(torch.tensor(1).repeat(1, n_antennas), offset=0)
for n in range(1,n_antennas):
    diags[n,:,:] = torch.diag_embed(torch.tensor(1).repeat(1,n_antennas - n),offset=n) + torch.diag_embed(torch.tensor(1).repeat(1,n_antennas - n),offset=-n)
diags = diags[None,:,:,:]

angle_grid = torch.linspace(-math.pi,math.pi,1000)

angleX,angleY = torch.meshgrid(angle_grid,angle_grid)

X_m = torch.zeros(1000,1000,4)
X_m[:,:,1] = angleX
X_m[:,:,3] = angleY

X_m = X_m.reshape(-1,4)

C = compute_learned_cov(X_m[:,:2],X_m[:,2:])
print(C.size())
sol = checking_pd(C,n_antennas)
print(sol.bool())
print(X_m[sol.bool(),:])
X_pd = X_m[sol.bool(),:]

print(torch.sum(X_pd[:,1] != X_pd[:,3]))
matrix = np.array(X_pd[X_pd[:,1] != X_pd[:,3]])

scatter_matrix = np.zeros((X_pd.shape[0],2))
scatter_matrix[:,0] = X_pd[:,1]
scatter_matrix[:,1] = X_pd[:,3]

plt.scatter(scatter_matrix[:,0],scatter_matrix[:,1],linewidth=0.5)
plt.show()

np.savetxt('matrix.txt',matrix)

#####

n_antennas = 3
indices = torch.arange(n_antennas)
X, Y = torch.meshgrid(indices,indices)
X = X[None, :, :]
Y = Y[None, :, :]
diags = torch.zeros(n_antennas,n_antennas,n_antennas)
diags[0, :, :] = torch.diag_embed(torch.tensor(1).repeat(1, n_antennas), offset=0)
for n in range(1,n_antennas):
    diags[n,:,:] = torch.diag_embed(torch.tensor(1).repeat(1,n_antennas - n),offset=n) + torch.diag_embed(torch.tensor(1).repeat(1,n_antennas - n),offset=-n)
diags = diags[None,:,:,:]

angle_grid = torch.linspace(-math.pi,math.pi,80)

angleX1,angleX2,angleX3,angleX4 = torch.meshgrid(angle_grid,angle_grid,angle_grid,angle_grid)

X_m = torch.zeros(80,80,80,80,6)
X_m[:,:,:,:,1] = angleX1
X_m[:,:,:,:,2] = angleX3
X_m[:,:,:,:,4] = angleX2
X_m[:,:,:,:,5] = angleX4

X_m = X_m.reshape(-1,6)

C = compute_learned_cov(X_m[:,:3],X_m[:,3:])
print(C.size())
sol = checking_pd(C,n_antennas)
print(sol.bool())
print(X_m[sol.bool(),:])
X_pd = X_m[sol.bool(),:]

print(torch.sum(X_pd[:,1] != X_pd[:,3]))
matrix = np.array(X_pd[X_pd[:,1] != X_pd[:,3]])

scatter_matrix = np.zeros((matrix.shape[0],4))
scatter_matrix[:,0] = X_pd[:,1]
scatter_matrix[:,1] = X_pd[:,2]
scatter_matrix[:,2] = X_pd[:,4]
scatter_matrix[:,3] = X_pd[:,5]

new = scatter_matrix[:,:3][scatter_matrix[:,3] == -1.55091286]
print(new.shape)

plt.scatter(scatter_matrix[:,2],scatter_matrix[:,3])
plt.show()

#fig = plt.figure(figsize=(12, 12))
#ax = fig.add_subplot(projection='3d')
#ax.scatter(scatter_matrix[:4000,1],scatter_matrix[:4000,2],scatter_matrix[:4000,3],linewidth=0.5)
#plt.show()

#np.savetxt('matrix.txt',matrix)