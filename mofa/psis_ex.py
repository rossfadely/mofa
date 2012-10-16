
import numpy as np
import matplotlib.pyplot as pl

from mofa import *

np.random.seed(13)


d = np.random.randn(2,250)
d[0,:] *= 5
d[1,:] *= 1
d[1,:] += 0.5 * d[0,:]


D, N = d.shape
K, M = 1, 1

fig=pl.figure()
pl.plot(d[0,:],d[1,:],'ko',alpha=0.25)
mix = Mofa(d.T, K, M, lock_psis=False,PPCA=True)
im = mix.means

mix.run_em()
mix.plot_2d_ellipses(0,1,edgecolor='b')
pl.plot(mix.means[:,0],mix.means[:,1],'bx',ms=15,label='PPCA')
mix.covs[0] = np.dot(mix.lambdas[0],mix.lambdas[0].T)
mix.plot_2d_ellipses(0,1,edgecolor='g')
mix.covs[0] = np.diag(mix.psis[0])
mix.plot_2d_ellipses(0,1,edgecolor='m')

pl.legend()
pl.title(r'Data $(D, N) = ({0}, {1})$, Model $(K, M) = ({2}, {3}), L = {4}$'.format(D,N,K,M,mix.logLs.sum()))
pl.xlim(-10,10)
pl.ylim(-10,10)
fig.savefig('init_ppca.png')

fig=pl.figure()
pl.plot(d[0,:],d[1,:],'ko',alpha=0.25)
mix = Mofa(d.T, K, M, lock_psis=True,PPCA=False)
im = mix.means

mix.run_em()
mix.plot_2d_ellipses(0,1,edgecolor='b')
pl.plot(mix.means[:,0],mix.means[:,1],'bx',ms=15,label='Psi fixed')
mix.covs[0] = np.dot(mix.lambdas[0],mix.lambdas[0].T)
mix.plot_2d_ellipses(0,1,edgecolor='g')
mix.covs[0] = np.diag(mix.psis[0])
mix.plot_2d_ellipses(0,1,edgecolor='m')

pl.legend()
pl.title(r'Data $(D, N) = ({0}, {1})$, Model $(K, M) = ({2}, {3}), L = {4}$'.format(D,N,K,M,mix.logLs.sum()))
pl.xlim(-10,10)
pl.ylim(-10,10)
fig.savefig('init_fixedpsis.png')

fig=pl.figure()
pl.plot(d[0,:],d[1,:],'ko',alpha=0.25)
mix = Mofa(d.T, K, M, lock_psis=False,PPCA=False)
im = mix.means

mix.run_em()
mix.plot_2d_ellipses(0,1,edgecolor='b')
pl.plot(mix.means[:,0],mix.means[:,1],'bx',ms=15,label='Psi free')
mix.covs[0] = np.dot(mix.lambdas[0],mix.lambdas[0].T)
mix.plot_2d_ellipses(0,1,edgecolor='g')
mix.covs[0] = np.diag(mix.psis[0])
mix.plot_2d_ellipses(0,1,edgecolor='m')

pl.legend()
pl.title(r'Data $(D, N) = ({0}, {1})$, Model $(K, M) = ({2}, {3}), L = {4}$'.format(D,N,K,M,mix.logLs.sum()))
pl.xlim(-10,10)
pl.ylim(-10,10)
fig.savefig('init_free.png')
