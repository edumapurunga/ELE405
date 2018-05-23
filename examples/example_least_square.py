"""

Example of using Least Square (module_name.ls).

In this example, a reference model G will be used for generating 
output data in function of a cosine function. In order to add error to the 
process, a random signal will be summed to the output. Having an input and its
output data for a model, the least square method can be performed.

Model to be estimated:

    y(t) = G(q^-1)u(t) + H(q^-1)e(t)

    G(q^-1) = A/B, H(q^-1) = 1/A

    A(q^-1) = 1 -0.99q^(-1)
    B(q^-1) = 0.01q^(-1)

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
from sysid import ls
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

## Number of samples
N=1000

## Fill the coefficients of the model for A and B
a=np.array([1, -0.99])
b=np.array([0, 0.01])

## Input: Cosine function
t=np.arange(N)
u=np.cos(t/10)
## Output: Response of the model to a Cosine function (u2)
y=signal.lfilter(b,a,u)

## Apply error to the output signal obtained using the reference model
v=signal.lfilter([1.0],a,np.random.randn(N))*0.005
yr=y+v

## Estimate theta using Least Square method
theta=ls(1,0,u,yr,1)

## Fill arrays to store the estimated parameters
a = np.append([1], theta[0][0])
b = np.append([0], theta[1][0])

## Title
print("Example of using Least Square")

# Plot input vs output
print("Input and output data")
plt.figure()
plt.plot(t, u, t, yr)
plt.show()

print("Coefficients of A ([a1, a2, ...]")
print(a)

print("Coefficients of B ([b0, b1, b2, ...]")
print(b)