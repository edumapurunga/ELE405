"""

Example of using Extended Least Square (sysid.els).

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
from sysid import els
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

## Number of samples
N = 1000

## Filling coeffiecients of the model for A and B
a = np.array([1, -0.99])
b = np.array([0.01])
c = np.array([1, 0.5])

## Input: Sine function
t = np.arange(N)
u = np.sin(t/50.0)
## Output: Response of the model to a Sine function (u2)(G)
y0 = signal.lfilter(b, a, u)

## Apply error to the output signal obtained using the reference model(H)
v = signal.lfilter(c, a, np.random.randn(N))*0.05
y = y0+v

## Estimate theta using Least Square method
a, b, c = els(1, 0, 1, u, y, 0)

## Title
print("Example of using Extended Least Square")

# Plot input vs output
print("Input and output data")
plt.figure()
plt.plot(t, y, label='output with noise')
plt.plot(t, y0, label='output without noise')
plt.legend()
plt.show()

print("Coefficients of A ([a1, a2, ...]")
print(a)

print("Coefficients of B ([b0, b1, b2, ...]")
print(b)

print("Coefficients of C ([c1, c2, ...]")
print(c)