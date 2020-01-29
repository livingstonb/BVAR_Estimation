import numpy as np
from eye import eye
from scipy.linalg import cholesky

class Gibbs:
	def sample(self, var, prior, n_draws, n_burn):
		invsigma0 = prior.inv_variances
		means0 = prior.means
		sigma0 = prior.variances
		gamma0 = prior.gamma
		ols = var.coeffs
		sigma = var.vcov
		X = var.XX
		Y = var.YY
		XX = np.matmul(X.T, X)
		lags = var.lags
		T = var.n_obs
		n_vars = var.n_vars

		isig_mean_0 = np.matmul(invsigma0, means0)

		# Gibbs' sampler
		n_coeffs = means0.shape[0]
		self.coeff_draws = np.zeros((n_draws-n_burn,n_coeffs), order='F')
		self.vol_draws = np.zeros((n_draws-n_burn,n_vars*n_vars), order='F')
		self.n_vars = var.n_vars
		self.lags = var.lags

		invsigma = cholesky(sigma)
		invsigma = np.linalg.inv(invsigma)
		invsigma = np.matmul(invsigma, invsigma.T)

		ikeep = 0
		for i in range(n_draws):
			## Draw coefficients condl on volatility
			# Inverse of the covariance matrix of posterior dist
			post_variance_inv = invsigma0 + np.kron(invsigma, XX)

			# Cholesky decomposition of covariance of posterior dist
			tmp = (post_variance_inv + post_variance_inv.T) / 2
			tmp = cholesky(tmp)
			tmp = np.linalg.inv(tmp)
			tmp = np.matmul(tmp, tmp.T)
			tmp = (tmp + tmp.T) / 2
			post_variance_chol = cholesky(tmp).T

			# Mean of the posterior distribution
			tmp = np.kron(invsigma, XX)
			tmp = np.matmul(tmp, ols.reshape((-1,1), order='F'))
			post_mean = np.linalg.lstsq(
				post_variance_inv,
				isig_mean_0 + tmp,
				rcond=None)[0]

			check = False
			n_pts = n_vars * (lags * n_vars + 1)
			meye = eye((lags-1)*n_vars, lags*n_vars)

			while not check:
				draws = np.random.normal(size=(n_pts,1))
				draws = draws.reshape((n_pts,1), order='F')
				new_coeffs = post_mean + np.matmul(
					post_variance_chol, draws)
				new_coeffs = new_coeffs.reshape(
					(n_vars*lags+1,n_vars), order='F')
				MX = np.concatenate(
					(new_coeffs[0:lags*n_vars,:].T, meye),
					axis=0)
				evals = np.linalg.eigvals(MX)
				lbmax = np.max(np.absolute(evals))
				if lbmax < 1:
					check = True

			## Draw volatility conditional on new coeffs
			# Assumes an inverse Wishart distribution for sigma
			resids = Y - np.matmul(X, new_coeffs)
			resids = resids - resids.mean(axis=0)[np.newaxis,:]
			df = n_vars + 1 + T
			gamma1 = gamma0 + np.matmul(resids.T, resids)
			invgamma1 = np.linalg.inv(gamma1)
			invgamma1 = (invgamma1 + invgamma1.T) / 2

			z = np.random.normal(size=(df, n_vars))
			z = np.matmul(z, cholesky(invgamma1))

			invsigma = np.matmul(z.T, z)
			sigma = np.linalg.inv(invsigma)

			# Keep the draws if i >= n_burn
			if i >= n_burn:
				self.coeff_draws[ikeep,:] = new_coeffs.flatten(order='F')
				self.vol_draws[ikeep,:] = sigma.flatten(order='F')
				ikeep += 1