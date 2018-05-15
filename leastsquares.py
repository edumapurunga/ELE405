import math
import numpy as np
import matplotlib.pyplot as plt


M=500
N=500

a=1
b=0.9

u=np.ones((M))
y=np.ones((M))

T=np.zeros((N,2))

for j in range(0, N):



    for i in range(0, M-1):
        y[i+1]=a*u[i]+b*y[i]+np.random.randn(1)*0.5


    y=y


    phi=np.c_[u[1:M-1],y[1:M-1]]


    R=np.dot(phi.T,phi)
    S=np.dot(phi.T,y[2:M])
    theta=np.linalg.solve(R,S)

    T[j]=theta.T


print(theta)
#for i in range(0, M):

#    yr=y+np.random.randn(N, 1)
#    S=np.dot(z.T,yr)
#    theta=np.linalg.solve(R,S)
#    T[i]=theta.T


# ultimo grafico
#plt.plot(x,yr,'ro')
#yy=np.dot(z,theta)
#plt.plot(x,yr,'ro')
#plt.plot(x,yy)

plt.figure()
plt.plot(T[:,0],T[:,1],'ro')


for i in range(1,2):
    print(i)

