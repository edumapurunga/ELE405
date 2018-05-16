from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import metodos

M=100
N=1000

b=np.array([0, 0.01])
a=np.array([1, -0.99])

u1=np.ones(N)
y1=signal.lfilter(b,a,u1)

t = np.arange(N)
u2=np.cos(t/10)
#u2=np.random.randn(N)*10
y2=signal.lfilter(b,a,u2)

# Repete M experimentos
T1=np.zeros((M,2))
T2=np.zeros((M,2))

for i in range(0, M):

    v=signal.lfilter([1.0],a,np.random.randn(N))*0.05
    yr1=y1+v
    yr2=y2+v

    theta1=metodos.arx(u1,yr1)
    theta2=metodos.arx(u2,yr2)

    T1[i]=theta1.T
    T2[i]=theta2.T

# Plota figuras
plt.figure()
plt.plot(T1[:,0],T1[:,1],'ro')
plt.plot(T2[:,0],T2[:,1],'bo')



