
import numpy as np
import matplotlib.pyplot as pl

from scipy.cluster.vq import kmeans
from scipy.linalg import inv
from matplotlib.patches import Ellipse


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

        self.K = K 
        self.M = M 

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
        self.covs = np.zeros((self.K,self.D,self.D))
	self.update_covs()

        # Empty arrays to be filled
        self.rs   = np.empty((self.K,self.N))
        self.beta = np.zeros((self.K,self.M,self.D))
        self.latent  = np.zeros((self.K,self.M,self.N))
        self.latent_covs = np.zeros((self.K,self.M,self.M))

    def _E_step(self):

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
            self.latent_covs[k] = np.eye(self.M) - step

    def _M_step(self):
        # check copy issues, and that data etc are static!!

        for k in range(self.K):

            # means, lambdas
            latents_aug  = np.concatenate((self.latent[k],np.ones((1,self.N))))
            latents_augw = np.tile(self.rs[k],(self.M+1,1)) * latents_aug
            step = np.dot(self._data.T,latents_augw.T)

            tmp = np.dot(latents_aug,latents_augw.T)
            tmp[:self.M,:self.M] += np.sum(self.rs[k]) * self.latent_covs[k]
            tmp = np.dot(step,inv(tmp))

            self.means[k] = tmp[:,self.M]
            self.lam[k]   = tmp[:,:self.M]

            # psi
            step = np.sum(tmp * step, axis = 1) / np.sum(self.rs[k])
            step = np.dot(self._datasq.T,self.rs[k]).T - step
            self.psi[k] = step


            # amplitudes
            self.amps[k] = np.sum(self.rs[k]) / self.N

	self.update_covs()
        
    def update_covs(self):
	for k in range(self.K):
            self.covs[k] = np.dot(self.lam[k],self.lam[k].T) + np.diag(self.psi[k])

        
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
        # self.covs[k].shape == (D,D)
        sgn, logdet = np.linalg.slogdet(self.covs[k])
        if sgn <= 0:
            return -np.inf * np.ones(X.shape[0])

        # X1.shape == (N,D)
        X1 = X - self.means[k]

        # X2.shape == (N,D)
        X2 = np.linalg.solve(self.covs[k], X1.T).T

        p = -0.5 * np.sum(X1 * X2, axis=1)
	
        return -0.5 * np.log((2 * np.pi) ** (X.shape[1])) - 0.5 * logdet + p

    def _invert_cov(self,k):
        """
        Calculate inverse covariance of mofa or ppca model,
        using inversion lemma
        """
        # probable slight speed up if psi kept as 1D array
        psiI = inv(np.diag(self.psi[k]))
        lam  = self.lam[k]
        lamT = lam.T
        step = inv(np.eye(self.M) + np.dot(lamT,np.dot(psiI,lam)))
        step = np.dot(step,np.dot(lamT,psiI))
        step = np.dot(psiI,np.dot(lam,step))

	return psiI - step

    def plot_2d_ellipses(self,d1,d2):
	"""
	Make a 2D plot of the model projected onto axes
	d1 and d2.
	"""
	for k in range(self.K):
	    mean = self.means[k,(d1, d2)]
	    cov = self.covs[k][((d1, d2),(d1, d2)), ((d1, d1), (d2, d2))]
	    self.plot_2d_ellipse(mean, cov)

    def plot_2d_ellipse(self, mu, cov, ax=None, **kwargs):
	"""
	Plot the error ellipse at a point given it's covariance matrix.
	"""
	# some sane defaults
	facecolor = kwargs.pop('facecolor', 'none')
	edgecolor = kwargs.pop('edgecolor', 'k')

	x, y = mu
	U, S, V = np.linalg.svd(cov)
	theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
	ellipsePlot = Ellipse(xy=[x, y],
			      width=2 * np.sqrt(S[0]),
			      height=2 * np.sqrt(S[1]),
			      angle=theta,
		facecolor=facecolor, edgecolor=edgecolor, **kwargs)

	if ax is None:
	    ax = pl.gca()
	ax.add_patch(ellipsePlot)

