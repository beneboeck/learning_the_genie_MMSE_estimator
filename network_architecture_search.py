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


def trail_1_NAS():
    lr = torch.log(torch.tensor(1.5e-5)) + 0.7 * torch.randn(1)
    lr = np.array(torch.exp(lr) + 5e-6)
    n_layers = np.random.choice([4,5])
    n_conv = np.random.choice([0,1,2])
    n_fully = n_layers - n_conv
    kernel_size = np.random.choice([5,7,9])

    return lr,n_layers,n_conv,n_fully,kernel_size