
import numpy as np
import pyfits as pf
from mixtures_halfverbeek import *
from scipy.cluster.vq import kmeans
import pylab as pl

#f = pf.open('field_0.fits')
#d = f[0].data
#f.close()

np.random.seed(13)

d = np.random.randn(2,10)
d[0,:] *= 5
#d[0,:] += 20 

d[1,:] *= 1
#d[1,:] -= 20 

b = np.random.randn(2,10)
b[0,:] *= 5
b[0,:] -= 5 

b[1,:] *= 1
b[1,:] += 10 - 0.5 * b[0,:]

d = np.concatenate((d,b),axis=1)

#d = np.loadtxt('foo.dat')

#np.savetxt('simple.dat',d)

print 'dshape',d.shape



mix = Mofa(d.T,2,M=1)
#print mix.psi
#print mix.means

fig=pl.figure()
pl.plot(d[0,:],d[1,:],'bo',alpha=0.5)
for i in range(len(d[0,:])):
    pl.text(d[0,i],d[1,i],'{0}'.format(i))
fig.savefig('data.png')
pl.plot(mix.means[:,0],mix.means[:,1],'rx')
fig.savefig('kmeans.png')
mix.plot_2d_ellipses(0,1)
fig.savefig('initialization.png')


print mix.means

for i in range(1):

    mix._E_step()
    print mix.rs
    mix._M_step()
    mix.plot_2d_ellipses(0,1)
    fig.savefig('step{0}.png'.format(i+1))
    #print mix.amps
    #print mix.psi
    #print mix.means

#x = np.dot(mix.lam[0],mix.latent[0])
#x[0,:] += np.ones(mix.N) * mix.means[0][0]
#x[1,:] += np.ones(mix.N) * mix.means[0][1]
#x = np.random.multivariate_normal(mix.means[0],mix.cov[0],1000)
#pl.plot(x[:,0],x[:,1],'go',alpha=0.25)
#x = np.random.multivariate_normal(mix.means[1],mix.cov[1],1000)
#pl.plot(x[:,0],x[:,1],'ro',alpha=0.25)


#mix = MixtureModel(d,2,M=1,mixtype='mofa')
#print mix.means
#print mix.psi
#mix.run_em()
#print mix.means
#print mix.psi
#mix.run_expectation_mofa()
#mix.run_maximization_mofa()
#mix._loglike_total_mofa()

#m = mog.MixtureModel(4,d.T)
#m.run_kmeans()
#m.run_em()
#print m.rs.shape
