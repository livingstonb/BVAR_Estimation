import numpy as np
from eye import eye
from scipy.linalg import cholesky

def IRF(gibbs, Idv, freqs, MY, maxhor):
	n_cases = len(Idv)
	ny = MY.shape[0]

	vols = gibbs.vol_draws
	coeffs = gibbs.coeff_draws
	n_vars = gibbs.n_vars
	lags = gibbs.lags
	n_gibbs = VOL.shape[0]

	w_low = freqs[0]
	w_high = freqs[1]

	Qmat = np.zeros((n_cases,n_gibbs,n_vars), order='F')      # Impulse vector
	IRFsr = np.zeros((n_cases,n_gibbs,ny*nrep), order='F')     # contains IRFs
	SVTsr = np.zeros((n_cases,n_gibbs,ny), order='F')          # Short unr contribution (6-32 quarters)
	SVTlr = np.zeros((n_cases,n_gibbs,ny), order='F')          # Long run contribution (80-inf quarters)
	VD = np.zeros((n_cases,n_gibbs,ny*maxhor), order='F')   # standard FEV

	state_space_generator = StateSpaceVARGenerator(n_vars, lags, MY)

	for i in range(n_gibbs):
	    state_space_VAR = state_space_generator.create_state_space_VAR(
	    	coeffs[i,:], vols[i,:])

class StateSpaceVARGenerator:
	def __init__(self, n_vars, lags, my):
		self.zeros = np.zeros((n_vars*(lags-1),n_vars), order='F')
		tmp = np.eye(n_vars*(lags-1), order='F')
		self.eye = np.concatenate((tmp,self.zeros), axis=1)
		self.my = my
		self.sig = np.eye(n_vars)

	def create_state_space_VAR(self, coeffs, vols):
		tmp_shape = (self.n_vars*self.lags+1,self.n_vars)
		tmp = np.reshape(coeffs.T, tmp_shape, order='F')
		dyn = dyn[:,:self.n_vars*self.lags]

		S = np.reshape(vols, (self.n_vars,self.n_vars))
		S = cholesky(S).T

		ss_var = StateSpaceVAR()
		ss_var.mx = np.concatenate((dyn,self.eye), axis=0)
		ss_var.my = self.my
		ss_var.me = np.concatenate((S,self.zeros), axis=0)
		ss_var.sig = self.sig
		return ss_var

class StateSpaceVAR:
	def __init__(self):
		self.mx = None
		self.my = None
		self.me = None
		self.sig = None