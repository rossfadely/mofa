
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
        self._datasq = self._data ** 2.

        # Run K-means
        self.means = kmeans(data,self.K)[0]

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(K)
        self.amps /= np.sum(self.amps)

        # Randomly assign factor loadings
        self.lam = np.random.randn(self.K,self.D,self.M)

        # Set (high rank) variance to variance of all data
        # Do something approx. here for speed?
        self.psi = np.var(self._data) * np.ones((self.K,self.D))

        # Set initial cov
        self.cov = np.zeros((self.K,self.D,self.D))
        for k in range(self.K):
            self.cov[k] = np.dot(self.lam[k],self.lam[k].T) + \
                np.diag(self.psi[k])

        # Empty arrays to be filled
        self.rs   = np.empty((self.K,self.N))
        self.beta = np.zeros((self.K,self.M,self.D))
        self.latent  = np.zeros((self.K,self.M,self.N))
        self.latent_cov = np.zeros((self.K,self.M,self.M))
        self.logL = np.zeros((self.K,self.N))
        self.logL_T = np.zeros(self.N)

    def _expectation(self):

        self._loglike()

        LogPxc = self.logL + np.tile(np.log(self.amps),(self.N,1)).T

        Q = LogPxc - np.tile(np.max(LogPxc,axis=0),(self.K,1))
        Q = np.exp(Q) / np.tile(np.sum(Q,axis=0),(self.K,1))

        #self.LogL_T = np.sum(Q * (LogPxc - np.log(Q))) 

    def _maximization(self):
        # check copy issues, and that data etc are static!!

        psisum = np.zeros((self.D,self.D))
        for k in range(self.K):

            # means, lambdas
            latents_aug  = np.concatenate((self.latent[k],np.ones((1,self.N))))
            latents_augw = np.tile(self.rs[k],(self.M+1,1)) * latents_aug
            step = np.dot(self._data.T,latents_augw.T)

            tmp = np.dot(latents_aug,latents_augw.T)
            tmp[:self.M,:self.M] += np.sum(self.rs[k]) * self.latent_cov[k]
            tmp = np.dot(step,inv(tmp))

            self.means[k] = tmp[:,self.M]
            self.lam[k]   = tmp[:,:self.M]

            # psi
            step = np.sum(tmp * step, axis = 1) / np.sum(self.rs[k])
            step = np.dot(self._datasq.T,self.rs[k]).T - step
            self.psi[k] = step


            # amplitudes
            self.amps[k] = np.sum(self.rs[k]) / self.N

        # Finish Psi
        #for k in range(self.K):
        #    self.cov[k] = np.dot(self.lam[k],self.lam[k].T) + np.diag(self.psi)

    def _loglike(self):

        for k in range(self.K):

            lam = self.lam[k].copy()
            psi = self.psi[k].copy()

            psiI  = np.atleast_2d(1./psi)
            psiIw = np.tile(psiI.T,(1,self.M)) * lam
            self.latent_cov[k] = inv(np.eye(self.M) + np.dot(lam.T,psiIw))

            psiI_w_xc = np.dot(psiIw.T,self._data.T) - \
                np.tile(np.dot(psiIw.T,self.means[k]),(self.N,1)).T

            self.latent[k] = np.dot(self.latent_cov[k],psiI_w_xc)

            sgn,ldet = np.linalg.slogdet(self.latent_cov[k])
            log_det = np.sum(np.log(psi)) - ldet

            energy = 0.5 * np.sum(psiI_w_xc * np.dot(self.latent_cov[k],psiI_w_xc),axis=0)
            energy = np.atleast_2d(energy)
            energy += np.dot((psiI * self.means[k]),self._data.T)
            energy -= 0.5 * np.dot(psiI,self.means[k] ** 2.)
            energy -= 0.5 * np.dot(psiI,self._datasq.T)

            self.logL[k] = energy - 0.5 * (log_det - self.D * np.log(2.*np.pi))
