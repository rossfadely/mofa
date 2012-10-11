
import numpy as np
import matplotlib.pyplot as pl

from mofa import *

np.random.seed(13)


d = np.random.randn(10,250)
d[0,:] *= 5
d[1,:] *= 1
d[1,:] += 3 + 0.5 * d[0,:]

b = np.random.randn(10,250)
b[0,:] *= 5
b[0,:] -= 5 
b[1,:] *= 1
b[1,:] += 10 - 0.5 * b[0,:]

d = np.concatenate((d,b),axis=1)

b = np.random.randn(10,130)
b[0,:] *= 1
b[0,:] += 5
b[1,:] *= 5
b[1,:] += 12 + 0.5 * b[0,:]

d = np.concatenate((d,b),axis=1)


fig=pl.figure()
pl.plot(d[0,:],d[1,:],'ko',alpha=0.25)

k,m = 2,3

mix = Mofa(d.T,k,m,False)
pl.plot(mix.means[:,0],mix.means[:,1],'rx',ms=15,label='Initialization')
mix.plot_2d_ellipses(0,1,edgecolor='r')

mix.run_em()

mix.plot_2d_ellipses(0,1,edgecolor='b')
pl.plot(mix.means[:,0],mix.means[:,1],'bx',ms=15,label='Psi free')

mix = Mofa(d.T,k,m,True)

mix.run_em()

mix.plot_2d_ellipses(0,1,edgecolor='g')
pl.plot(mix.means[:,0],mix.means[:,1],'gx',ms=15,label='Psi fixed')
pl.title('Data (K,D) = ({0},{1}), Model (K,M) = ({2},{3})'.format(4,5,k,m))
pl.legend()
fig.savefig('mofa_ex.png')

