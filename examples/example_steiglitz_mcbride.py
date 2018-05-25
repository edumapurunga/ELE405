"""

Example of using Steiglitz - Mc Bride (module_name.smb).

In this example, the Steiglitz-McBride method for estimating output-error (OE) models will be illustated.
For this class of algorithm, it will be considered a true system of OE type. 

Model to be estimated:

    y(t) = B(q^-1)/F(q^-1)u(t) + e(t)

    F(q^-1) = 1 - 1.2q^(-1) + 0.36q^(-1)
    B(q^-1) = 0.5 - 0.4^(-1)

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
from scipy import signal as sig
from sysid import steiglitz_mcbride

## Steiglitz-McBride Algorithm Example ##

#System Simulation#
#True System
Fo = [1, -1.2, 0.36] #Fo(q^-1) = 1 -1.2q^-1 + 0.36q^-2
Bo = [0.5, -0.4]     #Bo(q^-1) = 0.5 -0.4q^-1
thetao = np.array([Fo[1::], Bo])
#Number of Samples
N = 400; 
#Input
u = -1 + 2*np.random.rand(N, 1)
#u = np.ones((N,1))
ydet = np.zeros((N, 1))
y = np.zeros((N, 1))
#System orders
nf = 2; nb = 1; nk = 0;
#Signal-to-noise ratio
SNR = 20;

#Using filter command (OE structure)
ydet = sig.lfilter(Bo, Fo, u, 0);
#Adding some noise (defined accordingly to the specified SNR in dB)
stde = np.std(ydet)*10**(-SNR/20);

#Monte Carlo Simulation#

#Monte Carlo runs
MC = 100;
#Data storage
thetaSM = np.zeros((nf+nb+1, MC))
#Steigliz-McBride tolerance variables
tol = 1E-5
maxK = 50

#Monte Carlo runs
for i in range(0,MC):
    #New Noise Realization
    y = ydet + stde*np.random.randn(N, 1)
    #New estimative using the Steiglitz-McBride method 
    asm, bsm = steiglitz_mcbride(nf, nb, nk, tol, u, y)
    #Storage the new estimative
    thetaSM[0:nf, i:i+1] = asm
    thetaSM[nf::, i:i+1] = bsm

## Results

#bias
biasSM = la.norm(thetao.reshape(1, nf+nb+1)-np.mean(thetaSM, 1))
#covarince
covSM = np.cov(thetaSM)
varSM = la.norm(np.diag(covSM))

#Print the results
print('Results:\n')
print('Bias: %5.4f Variance: %5.4f.' % (biasSM, varSM))
