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

n_coherence = 10
N_train = 32000
N_val = 4000
N_test = 4000
n_paths = 3
n_antennas = 64

channel_generator = cg.SCMMulti(n_path = n_paths)

h, t_BS, t_MS = channel_generator.generate_channel(n_batches=N_train,n_coherence=n_coherence,n_antennas_BS=n_antennas,n_antennas_MS=1)


