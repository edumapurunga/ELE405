"""

Example of using Kalman Filter (sysid.kalman).

In this example, we have a state space model and we want to predict its states. A Kalman filter is used for that purporse.
As an example, we will consider the followint linear time-invariant(LTI) model:

#           x[k+1] = Ax[k] + Bu[k] + Gw[k]
#           y[k]   = Cx[k] + v[k]
#with
#
# A = |0.5  0.1|, B = |0.3| , C = [2 1] G = |0.6 0.1|^T
#     |-0.2 0.2|      |0.4|
#
#and w and v with the same variance equal to 1

Authors:
    augustomengarda
    cafeemymelo
    diegoeck
    edumapurunga
    gszzan
    max-feldman
    tulio.dapper

"""
import sys
sys.path.append('..')
import numpy as np
from numpy import linalg as la
from sysid import kalman
import matplotlib
import matplotlib.pyplot as plt

## Example of Kalman Algorithm ##

#Define the system in which we want to predict its states.
#System:
A = np.array([[0.5, 0.1],[-0.2, 0.2]])
B = np.array([.3, .4]).reshape((2, 1))
C = np.array([2, 1]).reshape((1, 2))
D = np.array(0)
G = np.array([.6, .1])
Q = np.array(1)
R = np.array(1)

n = 2 #Number of states

#Number of Samples
N = 300
#Choose an input
t = np.arange(0, N).reshape(N, 1) 
u = np.sin(0.1*t).reshape(N, 1)
#Chosen input
plt.plot(t, u)
#Simulate the system to get the measures
y = np.zeros((N, 1))     #Output
xdet = np.zeros((n, N))  #Determinsitic States
x = np.zeros((n, N))     #Noisy States
#Generate data
for i in range(0, N):
    #State update
    if i != N-1:
        xdet[:, i+1] = A.dot(xdet[:,i]) + B.dot(u[i])
        x[:, i+1] = xdet[:, i+1] + np.random.randn(1)
    y[i] = C.dot(x[:, i]) + np.random.randn(1)
#Guess of Initial Covariance Matrix
Po = np.eye(n)
xo = x[:,0:1]
#Call the Kalman filter function
K, Xkk, Vkk, Xkk1, Vkk1 = kalman(A, B, C, D, G, Q, R, u, y, xo, Po)
#K is the variable Kalman gain
#Xkk is the estimated corrected state
#Xkk1 is the estimated one-step predicted state
#Vkk is the estimated corrected error variance
#Vkk1 is the estimated one-step predicted error variance

#Results
#Print the Correction estimates
plt.figure(fign)
fign += 1 
plt.subplot(2, 1, 1)
plt.plot(t, x[0].T, 'o-', t, Xkk[0].T, '*-', t, xdet[0], 'k')
plt.title('Correction')
plt.ylabel('x1')
plt.subplot(2, 1, 2)
plt.plot(t, x[1].T, 'o-', t, Xkk[1].T, '*-', t, xdet[1], 'k' )
plt.xlabel('Samples')
plt.ylabel('x2')
plt.show()
#Print the Predition estimates
plt.figure(fign)
fign += 1
plt.subplot(2, 1, 1)
plt.plot(t, x[0].T, 'o-', t, Xkk1[0].T, '*-', t, xdet[0], 'k')
plt.title('Prediction')
plt.ylabel('x1')
plt.subplot(2, 1, 2)
plt.plot(t, x[1].T, 'o-', t, Xkk1[1].T, '*-', t, xdet[1], 'k')
plt.xlabel('Samples')
plt.ylabel('x2')
plt.show()
#Print the Kalman gain evolution
plt.figure(fign)
fign += 1
plt.subplot(2, 1, 1)
plt.plot(t, K[0].T, 'o-')
plt.title('Kalman Filter Gain')
plt.ylabel('x1')
plt.subplot(2, 1, 2)
plt.plot(t, K[1].T, 'o-')
plt.xlabel('Samples')
plt.ylabel('x2')
plt.show()