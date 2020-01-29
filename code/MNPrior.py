import numpy as np

class MNPrior:
	def __init__(self, data, AR, lags, params=[0.2,0.5,2.0,1e5]):
		self.T, self.n_vars = data.shape
		self.AR = AR
		self.hparam1 = params[0]
		self.hparam2 = params[1]
		self.hparam3 = params[2]
		self.hparam4 = params[3]

		self.compute(data, lags)

	def compute(self, data, lags):
		if self.AR:
			rho = np.asarray(self.AR)
		else:
			rho = np.ones(self.n_vars, order='F')

		se = np.zeros(self.n_vars, order='F')

		# Estimate error variances from OLS residuals
		for ivar in range(self.n_vars):
			y = data[1:self.T,ivar]
			x_noconst = data[0:self.T-1,ivar].reshape((-1,1), order='F')
			const = np.ones((self.T-1,1), order='F')
			x = np.concatenate((const,x_noconst), axis=1)
			b = np.linalg.lstsq(x, y, rcond=None)[0]
			u = y - np.matmul(x, b)
			se[ivar] = np.sqrt(u.dot(u) / (y.shape[0]-2))

		# Set coefficient means
		self.coeff_mean = np.zeros((lags,self.n_vars,self.n_vars), order='F')
		lag = 1
		for ivar in range(self.n_vars):
			self.coeff_mean[lag-1,ivar,ivar] = rho[ivar]

		# Set coefficient variances
		self.coeff_var = np.zeros((lags,self.n_vars,self.n_vars), order='F')
		for lag in range(1, lags+1):
			for ivar in range(self.n_vars):
				for jvar in range(self.n_vars):
					if ivar == jvar:
						tmp = self.hparam1 / (lag ** self.hparam3)
					else:
						tmp_denom = se[jvar] * lag ** self.hparam3
						tmp_num = se[ivar] * self.hparam1 * self.hparam2
						tmp = tmp_num / tmp_denom
					self.coeff_var[lag-1,ivar,jvar] = tmp * tmp

		# Set constant variances
		self.const_var = np.zeros(self.n_vars, order='F')
		for ivar in range(self.n_vars):
			tmp = se[ivar] * self.hparam4
			self.const_var[ivar] = tmp * tmp

		# Create matrix of all means
		self.means = np.zeros((lags*self.n_vars+1, self.n_vars), order='F')
		tmp = np.transpose(self.coeff_mean, (2,0,1))
		tmp = np.reshape(tmp, (lags*self.n_vars, self.n_vars), order='F')
		self.means[:-1,:] = tmp
		self.means = self.means.reshape((-1,1), order='F')
		
		# Create matrix of all variances
		self.variances = np.zeros((lags*self.n_vars+1, self.n_vars), order='F')
		tmp = np.transpose(self.coeff_var, (2,0,1))
		tmp = np.reshape(tmp, (lags*self.n_vars, self.n_vars), order='F')
		self.variances[:-1,:] = tmp
		self.variances[-1,:] = self.const_var
		self.variances = np.diagflat(self.variances.reshape((-1,1), order='F'))


		# WHAT IS THIS???
		self.gamma = np.eye(self.n_vars, order='F')

		n_stacked = self.n_vars * (self.n_vars * lags + 1)
		tmp = np.eye(n_stacked, order='F')

		self.inv_variances = np.linalg.lstsq(self.variances, tmp, rcond=None)[0]
		# self.coeff_var_inv = np.linalg.inv(self.coeff_var)
