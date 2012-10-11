import numpy as np
import matplotlib.pyplot as pl
import time

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
    `data`:        (N,D) array of observations
    `latents`:     (K,M,N) array of latent variables
    `latent_covs`: (K,M,M,N) array of latent covariances
    `lambdas`:     (K,M,D) array of loadings
    `psis`:        (K,D) array of diagonal variance values
    `rs`:          (K,N) array of responsibilities
    `amps`:        (K) array of component amplitudes

    """
    def __init__(self,data,K,M,lock_psis=False):

        self.K = K 
        self.M = M 

        self.data = np.atleast_2d(data)
        self.dataT = self.data.T # INSANE DATA DUPLICATION
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]

	self.lock_psis = lock_psis

        # Run K-means
        self.means = kmeans(data,self.K)[0]

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(K)
        self.amps /= np.sum(self.amps)

        # Randomly assign factor loadings
        self.lambdas = np.zeros((self.K,self.D,self.M))

        # Set (high rank) variance to variance of all data
        # Do something approx. here for speed?
        self.psis = np.tile(np.var(self.data,axis=0)[None,:],(self.K,1))
                            
        # Set initial covs
        self.covs = np.zeros((self.K,self.D,self.D))
	self._update_covs()

        # Empty arrays to be filled
        self.rs   = np.empty((self.K,self.N))
        self.betas = np.zeros((self.K,self.M,self.D))
        self.latents  = np.zeros((self.K,self.M,self.N))
        self.latent_covs = np.zeros((self.K,self.M,self.M,self.N))

    def run_em(self, maxiter=400, tol=1e-4, verbose=True):
        """
        Run the EM algorithm.

        :param maxiter:
            The maximum number of iterations to try.

        :param tol:
            The tolerance on the relative change in the loss function that
            controls convergence.

        :param verbose:
            Print all the messages?

        """

        L = None
        for i in xrange(maxiter):
            self._E_step()
            newL = self.logLs.sum()
            if i == 0 and verbose:
                print("Initial NLL =", -newL)

            self._M_step()
            if L is None:
                L = newL
            else:
                dL = np.abs((newL - L) / L)
                if i > 5 and dL < tol:
                    break
                L = newL

        if i < maxiter - 1:
            if verbose:
                print("EM converged after {0} iterations".format(i))
                print("Final NLL = {0}".format(-newL))
        else:
            print("Warning: EM didn't converge after {0} iterations"
                    .format(i))
    
    def take_EM_step(self):

        self._E_step()
        self._M_step()


    def _E_step(self):
        """
        Expectation step.  See docs for details.
        """
        # resposibilities and likelihoods
        self.logLs, rs = self._calc_prob()
        self.rs = rs.T

        for k in range(self.K):
            # beta
            invcov = self._invert_cov(k)
            self.betas[k] = np.dot(self.lambdas[k].T,invcov)

            # latent values
            zeroed = self.dataT - self.means[k, :, None]
            self.latents[k] = np.dot(self.betas[k],zeroed)

            # latent cov
	    step1   = self.latents[k, :, None, :] * self.latents[k, None, :, :]
            step2   = np.dot(self.betas[k],self.lambdas[k])
	    self.latent_covs[k] = np.eye(self.M)[:,:,None] - step2[:,:,None] + step1

    def _M_step(self):
        """
        Maximization step.  See docs for details.

        This assumes that `_E_step()` has been run.
        """
	sumrs = np.sum(self.rs,axis=1)
        for k in range(self.K):

            # means
            lambdalatents = np.dot(self.lambdas[k], self.latents[k])
            self.means[k] = np.sum(self.rs[k] * (self.dataT - lambdalatents),
                                   axis=1) / sumrs[k]

            # lambdas
            zeroed = self.dataT - self.means[k, :, None]
	    self.lambdas[k] = np.dot(np.dot(zeroed[:,None,:] * self.latents[k,None,:,:],
                                            self.rs[k]),
                                     inv(np.dot(self.latent_covs[k],
                                                self.rs[k])))

            # psis - not this is not in any paper MOFAAAAA!
	    self.psis[k] = np.dot((zeroed - lambdalatents) *
                                  zeroed,self.rs[k]) / sumrs[k]

            # amplitudes
            self.amps[k] = sumrs[k] / self.N

	if self.lock_psis:
	    psi = np.dot(sumrs, self.psis) / np.sum(sumrs)
	    for k in range(self.K):
		self.psis[k] = psi

	self._update_covs()

    def _update_covs(self):
        """
        Update self.cov for responsibility, logL calc
        """
	for k in range(self.K):
            self.covs[k] = np.dot(self.lambdas[k],self.lambdas[k].T) + \
		np.diag(self.psis[k])
        
    def _calc_prob(self):
        """
        Calculate log likelihoods, responsibilites for each datum
        under each component.
        """
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
        """
        Gaussian log likelihood of the data for component k.
        """
        sgn, logdet = np.linalg.slogdet(self.covs[k])
        assert sgn > 0

        X1 = X - self.means[k]
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
	    self._plot_2d_ellipse(mean, cov, **kwargs)

    def _plot_2d_ellipse(self, mu, cov, ax=None, **kwargs):
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

    def lnprob(self, x):
        """
        Alias for calculating log likelihoods
        """
        return self._calc_prob(x)[0]
