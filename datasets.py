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
from torch.utils.data import Dataset, DataLoader
import math
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import network_architecture_search as nas
import networks as n

class dataset_trial1(Dataset):
    def __init__(self,h,C_sim):
        super().__init__()
        h = torch.tensor(h)
        C_sim = torch.tensor(C_sim)
        n_coherence = h.size(1)
        C_hat = 1/(n_coherence - 1) * torch.einsum('ijh,ijk->ijhk',h,torch.conj(h)).sum(dim=1)
        self.C_hat = torch.zeros(C_hat.size(0),2,C_hat.size(1),C_hat.size(2))
        self.C_hat[:,0,:,:] = torch.real(C_hat)
        self.C_hat[:,1,:,:] = torch.imag(C_hat)
        self.C_sim = C_sim


    def __len__(self):
        return self.C_sim.size(0)

    def __getitem__(self,idx):
        return self.C_hat[idx,:,:,:],self.C_sim[idx,:,:]