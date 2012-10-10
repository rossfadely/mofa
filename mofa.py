
import numpy as np
import matplotlib.pyplot as pl

from scipy.cluster.vq import kmeans
from scipy.linalg import inv
from matplotlib.patches import Ellipse


class Mofa(object):
    """
    Mixture of Factor Analyzers
    
    `K`:           Number of components
    `M`:           Latent dimensionality
    `D`:           Data dimensionality
    `N`:           Number of data points
    `data`:        (D,N) array of observations
    `latents`:     (K,M,N) array of latent variables
    `latent_covs`: (K,M,M,N) array of latent covariances
    `lambdas`:     (K,M,D) array of loadings
    `psis`:        (K,D) array of diagonal variance values
    `rs`:          (K,N) array of responsibilities
    `amps`:        (K) array of component amplitudes

    TODO - LogL and convergence
    """
    def __init__(self,data,K,M,lock_psis=False):

        self.K = K 
        self.M = M 

        self.data = np.atleast_2d(data)
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]

	self.lock_psis = lock_psis

        # Run K-means
        self.means = kmeans(data,self.K)[0]

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(K)
        self.amps /= np.sum(self.amps)

        # Randomly assign factor loadings
        self.lambdas = np.random.randn(self.K,self.D,self.M)

        # Set (high rank) variance to variance of all data
        # Do something approx. here for speed?
        self.psis = np.var(self.data) * np.ones((self.K,self.D))

        # Set initial cov
        self.covs = np.zeros((self.K,self.D,self.D))
	self._update_covs()

        # Empty arrays to be filled
        self.rs   = np.empty((self.K,self.N))
        self.betas = np.zeros((self.K,self.M,self.D))
        self.latents  = np.zeros((self.K,self.M,self.N))
        self.latent_covs = np.zeros((self.K,self.M,self.M,self.N))

    def _E_step(self):

        # resposibilities and likelihoods
        self.logL, rs = self._calc_prob()
        self.rs = rs.T

        for k in range(self.K):
            # beta
            invcov = self._invert_cov(k)
            self.betas[k] = np.dot(self.lambdas[k].T,invcov)

            # latent values
            zeroed = (self.data - self.means[k]).T
            self.latents[k] = np.dot(self.betas[k],zeroed)

            # latent cov
	    step1   = self.latents[k, :, None, :] * self.latents[k, None, :, :]
            step2   = np.dot(self.betas[k],self.lambdas[k])
	    self.latent_covs[k] = np.eye(self.M)[:,:,None] - step2[:,:,None] + step1

    def _M_step(self):

	sumrs = np.sum(self.rs,axis=1)
        for k in range(self.K):

            # means
            step = self.data.T - np.dot(self.lambdas[k],self.latents[k])
            self.means[k] = np.sum(self.rs[k] * step,axis=1) / sumrs[k]
 
            # lambdas
            zeroed = (self.data - self.means[k]).T
	    right  = inv(np.dot(self.latent_covs[k],self.rs[k]))
	    left  = np.dot(zeroed[:,None,:]*self.latents[k,None,:,:],self.rs[k])
	    self.lambdas[k] = np.dot(left,right)

            # psi - not this is not in any paper MOFAAAAA!
	    ddT  = zeroed[:,None,:] * zeroed[None,:,:]
	    step = np.dot(self.lambdas[k],self.latents[k])[:,None,:] * zeroed[None,:,:]
	    self.psis[k] = np.diag(np.dot(ddT-step,self.rs[k]) / sumrs[k])

            # amplitudes
            self.amps[k] = sumrs[k] / self.N

	
	if self.lock_psis:
	    psi = np.dot(sumrs, self.psis) / np.sum(sumrs)
	    for k in range(self.K):
		self.psis[k] = psi

	self._update_covs()
        
    def _update_covs(self):
	for k in range(self.K):
            self.covs[k] = np.dot(self.lambdas[k],self.lambdas[k].T) + \
		np.diag(self.psis[k])

        
    def _calc_prob(self):

        logrs = []
        for k in range(self.K):
            logrs += [np.log(self.amps[k]) + self._log_multi_gauss(k, self.data)]
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
        psiI = inv(np.diag(self.psis[k]))
        lam  = self.lambdas[k]
        lamT = lam.T
        step = inv(np.eye(self.M) + np.dot(lamT,np.dot(psiI,lam)))
        step = np.dot(step,np.dot(lamT,psiI))
        step = np.dot(psiI,np.dot(lam,step))

	return psiI - step

    def plot_2d_ellipses(self,d1,d2, **kwargs):
	"""
	Make a 2D plot of the model projected onto axes
	d1 and d2.
	"""
	for k in range(self.K):
	    mean = self.means[k,(d1, d2)]
	    cov = self.covs[k][((d1, d2),(d1, d2)), ((d1, d1), (d2, d2))]
	    self.plot_2d_ellipse(mean, cov, **kwargs)

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

