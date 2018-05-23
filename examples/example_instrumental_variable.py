"""

Example of using Instrumental Variable (module_name.iv).

In this example, a reference model G will be used for generating output data. 
Two experiments will be simulated by generating two uncorrelated random signals 
applied the output from the model with the same input signal.

Model to be estimated:

    y(t) = G(q^-1)u(t) + H(q^-1)e(t)

    G(q^-1) = A/B, H(q^-1) = unknown

    A(q^-1) = 1 -0.99q^(-1)
    B(q^-1) = 5q^(-1)

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
from sysid import iv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

## Number of samples in each experiment
N=1000

## Fill the coefficients of the model for A and B
a=np.array([1, -0.99])
b=np.array([0, 5])

## Input: Sine function
t=np.arange(N)
u=np.sin(t/10)
## Output: Response of the model to a Sine function (u1)
y=signal.lfilter(b,a,u)

## Apply error to the output signal obtained using the reference model
v=np.random.normal(0,7,N)
yr1=y+v
v=np.random.normal(0,7,N)
yr2=y+v

## Estimate theta using Instrumental Variable method
theta=iv(1,0,u,yr1,yr2,1)

## Fill arrays to store the estimated parameters
a = np.append([1], theta[0][0])
b = np.append([0], theta[1][0])

## Title
print("Example of using Instrumental Variable")

# Plot input vs output
print("Input and output data")
plt.figure()
plt.plot(t, u, t, yr1, t, yr2)
plt.show()

print("Coefficients of A ([a1, a2, ...]")
print(a)

print("Coefficients of B ([b0, b1, b2, ...]")
print(b)