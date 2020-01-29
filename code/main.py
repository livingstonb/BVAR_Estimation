from scipy.io import loadmat
from toolbox import minnesota_priors
from MNPrior import MNPrior
from VAR import VAR
from Gibbs import Gibbs
from IRF import IRF
import numpy as np

lags = 2
T0 = 1955
T1 = 2017.75

# Read and prepare data
data = loadmat("data_1955_2017.mat")
data = data["X"]
time = data[:,0]
data = data[(T0<=time) & (time<=T1),1:]
data = data.reshape(data.shape, order='F')

# Priors
AR = []
mn_prior = MNPrior(data, AR, lags)

# Estimate VAR
rvar = VAR()
rvar.estimate(data, lags)

# Perform Gibbs' sampling
n_draws = 1000
n_burn = 900
gibbs = Gibbs()
gibbs.sample(rvar, mn_prior, n_draws, n_burn)

# IRF
maxhor = 200        # Maximal horizon when testing for LR effects
nrep = 40           # Number of periods 
test = 1            # Testing period for the IRF

freqs = [2*np.pi/32, 2*np.pi/6]
Idv = [1,2,4,5]    # index of Variable for which we maximize the variance contribution
                       # Here: 1->Y, 2->I, 3->h, 4->u
n_vars = rvar.n_vars
Id   = np.eye(n_vars, order='F')
zrs = np.zeros((n_vars,n_vars*(lags-1)), order='F')
MYtmp = np.concatenate((Id, zrs), axis=1)

# y i c h u sw R pi y/h TFP 
# 0 1 2 3 4 5  6 7  8   9
MY = MYtmp[0:10,:]

IRF(gibbs, Idv, freqs, MY, maxhor)