"""

Example of using Instrumental Variable (module_name.iv).

In this example, a reference model G will be used for generating output data. 
Two experiments will be simulated by generating two uncorrelated random signals 
applied the output from the model with the same input signal.

Model to be estimated:

    y(t) = G(q^-1)u(t) + H(q^-1)e(t)

    G(q^-1) = A/B, H(q^-1) = unknown

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

## Number of samples in each experiment
N=1000

## Fill the coefficients of the model for A and B
a0=np.array([1, -0.99])
b0=np.array([0.01])

## Input: Sine function
t=np.arange(N)
u=np.sin(t/50)
## Output: Response of the model to a Sine function (u1)
y0=signal.lfilter(b0,a0,u)

## Apply error to the output signal obtained using the reference model
v=np.random.randn(N)*0.05
y1=y0+v
v=np.random.randn(N)*0.05
y2=y0+v

## Estimate theta using Instrumental Variable method
a,b=iv(1,0,u,y1,y2,0)

## Simulation
ys=signal.lfilter(b,np.array([1, a]),u)

## Title
print("Example of using Instrumental Variable")

# Plot input vs output
print("Input and output data")
plt.figure()
plt.plot(t, y1 , label='output with noise')
plt.plot(t, y0, label='output without noise')
plt.plot(t, ys, label='simulation')
plt.legend()

print("Coefficients of A ([a1, a2, ...]")
print(a)

print("Coefficients of B ([b_k, b_k+1, b_k+2, ...]")
print(b)