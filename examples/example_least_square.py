"""

Example of using Least Square (module_name.ls).

In this example, a reference model G will be used for generating 
output data in function of a cosine function. In order to add error to the 
process, a random signal will be summed to the output. Having an input and its
output data for a model, the least square method can be performed.

Model to be estimated:

    y(t) = G(q^-1)u(t) + H(q^-1)e(t)

    G(q^-1) = A/B, H(q^-1) = 1/A

    A(q^-1) = 1 - 0.99q^(-1)
    B(q^-1) = 0.01

Authors:
    augustomengarda
    cafeemymelo
    diegoeck
    edumapurunga
    gszzan
    max-feldman
    tulio.dapper

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from sysid import ls
from scipy import signal

## Number of samples
N=1000

## Fill the coefficients of the model for A and B
a0=np.array([1, -0.99])
b0=np.array([0.01])

## Input: Cosine function
t=np.arange(N)
u=np.sin(t/50)
## Output: Response of the model to a Cosine function (u)
y0=signal.lfilter(b0,a0,u)

## Apply error to the output signal obtained using the reference model
v=signal.lfilter([1.0],a0,np.random.randn(N))*0.05
y=y0+v

## Estimate theta using Least Square method
print("Example of using Least Square")
a,b=ls(1,0,0,u,y)

## Simulation
ys=signal.lfilter(b,np.array([1, a]),u)

## Predicition
yp=signal.lfilter(b,1,u)+signal.lfilter(np.array([0, -a]),1,y)

# Plot input vs output
print("Input and output data")
plt.figure()
plt.plot(t, y , label='output with noise')
plt.plot(t, y0, label='output without noise')
plt.plot(t, ys, label='simulation')
plt.plot(t, yp, label='prediction')
plt.legend()

print("Coefficients of A ([a1, a2, ...]")
print(a)

print("Coefficients of B ([b_k, b_k+1, b_k+2, ...]")
print(b)