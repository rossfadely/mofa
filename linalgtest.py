import numpy as np


rs = np.arange(200) - 350
rs = [2,10]

b  = np.arange(4*10).reshape(4,10)

# method A

d = np.zeros(10)

for i in range(2):
    c = np.zeros(10)
    for j in range(4):

        c += b[j]
        
    d += rs[i] * c

print d

e = np.zeros(10)
f = np.zeros(10)
for j in range(4):
    for i in range(2):
        f += rs[i] * b[j]

print f




# method B

for j in range(1):

    tmp = b[j].flatten()
    for i in range(2):
        tmp[i] += np.sum(rs[0] * tmp[i])

e = tmp.reshape(10,10)

print d-e
