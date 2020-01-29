import numpy as np
from eye import eye

class VAR:
    def estimate(self, X, lags, constant=True):
        self.n_obs, self.n_vars = X.shape
        self.lags = lags
        self.YY, self.XX = buildmat(X, self.lags, constant)
        T = self.XX.shape[0]

        self.coeffs = np.linalg.lstsq(self.XX, self.YY, rcond=None)[0]
        n_coeffs = self.coeffs.shape[0]

        XXprod = np.matmul(self.XX.T, self.XX)
        # XXinv = np.linalg.inv(XXprod)
        XXinv = np.linalg.lstsq(XXprod, np.eye(n_coeffs, order='F'), rcond=None)[0]
        self.Yfit = np.matmul(self.XX, self.coeffs)
        self.resid = self.YY - self.Yfit
        self.SSR = np.matmul(self.resid.T, self.resid)
        self.vcov = self.SSR / (T-self.lags)

        ldet = np.log(np.linalg.det(np.cov(self.resid.T)))
        n_bc = self.coeffs.size
        self.aic = ldet + 2 * n_bc / self.n_obs
        self.bic = ldet + n_bc * np.log(self.n_obs) / self.n_obs
        self.hq = ldet + 2 * n_bc * np.log(np.log(self.n_obs)) / self.n_obs

        self.VX = np.kron(self.vcov, XXinv)
        self.SX = np.sqrt(np.diag(self.VX))
        self.SX = self.SX.reshape((n_coeffs, self.n_vars), order='F')

        tmp1 = self.coeffs[0:self.lags*self.n_vars,:].T
        tmp2 = eye((self.lags-1)*self.n_vars, self.lags*self.n_vars)
        self.SScoeffs = np.concatenate((tmp1, tmp2), axis=0)

def buildmat(X, lags, constant=True):
    T, n_vars = X.shape

    XX = np.zeros((T-lags,n_vars,lags), order='F')
    for lag in range(1, lags+1):
        XX[:,:,lag-1] = np.reshape(X[lags-lag:T-lag,:], (T-lags,n_vars), order='F')
    XX = XX.reshape((T-lags,lags*n_vars), order='F')

    if constant:
        XX = np.concatenate((XX, np.ones((T-lags,1), order='F')), axis=1);

    YY = np.reshape(X[lags:T,:], (T-lags,n_vars), order='F')

    return YY, XX