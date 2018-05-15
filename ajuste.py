import math
import numpy as np
import matplotlib.pyplot as plt


M=50
N=50


t0=np.array([[1,1]]).T

def phi1(x):
    z=np.c_[np.power(x,2),np.power(x,1)]
    return z

def phi2(x):
    z=np.c_[np.power(x,2),np.power(x,1)]
    return z

x=np.arange(1,N+1)
y=np.dot(phi1(x),t0)




T = np.zeros(shape=(M,2))

z=phi2(x)
R=np.dot(z.T,z)


for i in range(0, M):

    yr=y+np.random.randn(N, 1)
    S=np.dot(z.T,yr)
    theta=np.linalg.solve(R,S)
    T[i]=theta.T


# ultimo grafico
plt.plot(x,yr,'ro')
yy=np.dot(z,theta)
plt.plot(x,yr,'ro')
plt.plot(x,yy)

plt.figure()
plt.plot(T[:,0],T[:,1],'ro')




