
import numpy as np

from scipy.cluster.vq import kmeans
from scipy.linalg import inv

import time

__doc__ = """

Define -

K: Number of components
M: Latent dimensionality
data: (D,N) array of observations
latent:  (K,N) array of latent variables
lam:  (K,M,D) array of loadings
amps: (K,1) array of component amplitudes 

"""

class Mofa(object):
    """

    TODO - add checks on inputs
    """

    def __init__(self,data,K,M=None):

        self.K    = K 
        self.M    = M 

        self._data = np.atleast_2d(data)
        self.N = self._data.shape[0]
        self.D = self._data.shape[1]
        self._datadataT = np.dot(self._data,self._data.T)

        # Run K-means
        self.means = kmeans(data,self.K)[0]

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(K)
        self.amps /= np.sum(self.amps)

        # Randomly assign factor loadings
        self.lam = np.random.randn(self.K,self.D,self.M)

        # Set (high rank) variance to variance of all data
        # Do something approx. here for speed?
        self.psi = np.var(self._data) * np.ones(self.D)

        # Set initial cov
        self.cov = np.zeros((self.K,self.D,self.D))
        for k in range(self.K):
            self.cov[k] = np.dot(self.lam[k],self.lam[k].T) + \
                np.diag(self.psi)

        # Empty arrays to be filled
        self.rs   = np.empty((self.K,self.N))
        self.beta = np.zeros((self.K,self.M,self.D))
        self.latent  = np.zeros((self.K,self.M,self.N))
        self.latent_cov = np.zeros((self.K,self.M,self.M))

    def _expectation(self):

        # resposibilities and likelihoods
        self.logL, rs = self._calc_prob()
        self.rs = rs.T

        for k in range(self.K):
            # beta
            invcov = self._invert_cov(k)
            self.beta[k] = np.dot(self.lam[k].T,invcov)

            # latent values
            zeroed = (self._data - self.means[k]).T
            self.latent[k] = np.dot(self.beta[k],zeroed)

            # latent cov
            step   = np.dot(zeroed,np.dot(zeroed.T,self.beta[k].T))
            step   = np.dot(self.beta[k],self.lam[k]) + np.dot(self.beta[k],step)
            self.latent_cov[k] = np.eye(self.M) - step

    def _maximization(self):
        # check copy issues, and that data etc are static!!

        psisum = np.zeros((self.D,self.D))
        for k in range(self.K):

            # means
            step = self._data.T - np.dot(self.lam[k],self.latent[k])
            self.means[k] = np.sum(self.rs[k] * step,axis=1) / self.rs[k].sum()
 
            # lambda
            zeroed = self._data - self.means[k]
            right = self.latent_cov[k].ravel().copy()
            for i in range(self.M**2):
                right[i] = np.sum(self.rs[k] * right[i])
            right = inv(right.reshape(self.M,self.M))

            left = np.dot(zeroed.T,self.latent[k].T).ravel()
            for i in range(self.D*self.M):
                left[i] = np.sum(self.rs[k] * left[i])
            self.lam[k] = np.dot(left.reshape(self.D,self.M),right)

            # psi (or zeroed?)
            ddT  = np.dot(self._data.T,self._data)
            step = np.dot(self.lam[k],np.dot(self.latent[k],self._data))
            step = ddT - step
            step = step.ravel()
            for i in range(self.D**2):
                step[i] = np.sum(self.rs[k] * step[i])
            psisum += step.reshape(self.D,self.D)
            

            # amplitudes
            self.amps[k] = np.sum(self.rs[k]) / self.N

        # Finish Psi
        self.psi = np.diag(psisum) / self.N
        for k in range(self.K):
            self.cov[k] = np.dot(self.lam[k],self.lam[k].T) + np.diag(self.psi)

        
    def _calc_prob(self):

        logrs = []
        for k in range(self.K):
            logrs += [np.log(self.amps[k]) + self._log_multi_gauss(k, self._data)]
        logrs = np.concatenate(logrs).reshape((-1, self.K), order='F')

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        a = np.max(logrs, axis=1)
        L = a + np.log(np.sum(np.exp(logrs - a[:, None]), axis=1))
        logrs -= L[:, None]
        return L, np.exp(logrs)

        
    def _log_multi_gauss(self, k, X):
        # X.shape == (N,D)
        # self.means.shape == (D,K)
        # self.cov[k].shape == (D,D)
        sgn, logdet = np.linalg.slogdet(self.cov[k])
        if sgn <= 0:
            print self.psi
            print np.dot(self.lam,self.lam.T)
            return -np.inf * np.ones(X.shape[0])

        # X1.shape == (N,D)
        X1 = X - self.means[k]

        # X2.shape == (N,D)
        X2 = np.linalg.solve(self.cov[k], X1.T).T

        p = -0.5 * np.sum(X1 * X2, axis=1)

        out = -0.5 * np.log((2 * np.pi) ** (X.shape[1])) - 0.5 * logdet + p

        return out

    def _invert_cov(self,k):
        """
        Calculate inverse covariance of mofa or ppca model,
        using inversion lemma
        """
        # probable slight speed up if psi kept as 1D array
        psiI = inv(np.diag(self.psi))
        lam  = self.lam[k]
        lamT = lam.T
        step = inv(np.eye(self.M) + np.dot(lamT,np.dot(psiI,lam)))
        step = np.dot(step,np.dot(lamT,psiI))
        step = np.dot(psiI,np.dot(lam,step))

        return psiI - step

