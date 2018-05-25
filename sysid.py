# -*- coding: utf-8 -*-
"""
@authors:
    augustomengarda
    cafeemymelo
    diegoeck
    edumapurunga
    gszzan
    max-feldman
    tulio.dapper

"""

import numpy as np
import numpy.linalg as la
from scipy import signal as sig #Necessary for SM

def ls(na, nb, u, y, nk=0):

    # Number of samples
    N = np.size(y)

    # Vetor u and y must have same amount of samples
    if N != np.size(u):
        raise ValueError('Y and U must have same length!')

    # Number of coefficients to be estimated
    # (a_1, a_2, a_3,..., a_na, b_0, b_1, b_2, b_nb)
    M = na + nb + 1

    # Delay maximum needed
    n_max = np.amax([na, nb + nk])

    # In order to estimate the coeffiecients, we will need to delay the samples.
    # If the maximum order is greater than the number of samples,
    # then it will not be possible!
    if not (N - n_max > 0):
        raise ValueError('Number of samples should be greater' &
                         'than the maximum order!')

    # Build matrix phi in which will contain y and u shifted in time
    phi = np.zeros((N - n_max, M))

    k = 0

    # Fill phi with y shifted in time from 0 to nb
    for i in range(1, na + 1):
        phi[:, k] = -y[n_max - i:N - i]
        k = k + 1

    # Fill phi with u shifted in time from 0 to nb
    for i in range(nk, nb + nk + 1):
        phi[:, k] = u[n_max - i:N - i]
        k = k + 1


    # Crop y from n_max to N
    y = y[n_max:N]

    # Find theta
    R = np.dot(phi.T, phi)
    
    # If the experiment is not informative:
    if (la.matrix_rank(R) < M):
        raise ValueError('Experiment is not informative')

    S = np.dot(phi.T, y)
    theta = la.solve(R, S)

    # Split theta in vectors a and b
    a = theta[0:na]
    b = theta[na:na + nb + 1]

    return [a, b]


def iv(na, nb, u, y, y2, nk=0):

    # Number of samples
    N = np.size(y)

    # Vetors u, y and y2 must have same amount of samples
    if (N != np.size(u)) or (N != np.size(y2)):
        raise ValueError('Y, Y2 and U must have same length!')

    # Number of coefficients to be estimated
    # (a_1, a_2, a_3,..., a_na, b_0, b_1, b_2, b_nb)
    M = na + nb + 1

    # Delay maximum needed
    n_max = np.amax([na, nb + nk])

    # In order to estimate the coeffiecients, we will need to delay the samples.
    # If the maximum order is greater than the number of samples,
    # then it will not be possible!
    if not (N - n_max > 0):
        raise ValueError('Number of samples should be greater' &
                         'than the maximum order!')

    # Build matrix phi in which will contain y and u shifted in time
    phi = np.zeros((N - n_max, M))

    # Build matrix csi in which will contain y2 and u shifted in time
    csi = np.zeros((N - n_max, M))

    k = 0

    # Fill phi/csi with y/y2 shifted in time from 0 to nb
    for i in range(1, na + 1):
        phi[:, k] = -y[n_max - i:N - i]
        csi[:, k] = -y2[n_max - i:N - i]
        k = k + 1

    # Fill phi/csi with u shifted in time from 0 to nb
    for i in range(nk, nb + nk + 1):
        phi[:, k] = u[n_max - i:N - i]
        csi[:, k] = u[n_max - i:N - i]
        k = k + 1


    # Crop y from n_max to N
    y = y[n_max:N]

    # Find theta
    R = np.dot(csi.T, phi)

    # If the experiment is not informative:
    if (la.matrix_rank(R) < M):
        raise ValueError('Experiment is not informative')

    S = np.dot(csi.T, y)
    theta = la.solve(R, S)

    # Split theta in vectors a and b
    a = theta[0:na]
    b = theta[na:na + nb + 1]

    return [a, b]

def steiglitz_mcbride(nf, nb, nk, tol, u, y):
    """
    This function implements the Steiglitz-Mcbride method for dynamic system 
    identification of single input single output (SISO) output-error models.
    Output error models are as following:

        Output Error (OE) model: y(k) = q^(-nk)B(q^-1)/F(q^-1)*u(k) + e(k) 
             
    Inputs:
        nf: Desired order of the polynomial F(q^-1) (scalar integer)
        nb: Desired order of the polynomial B(q^-1) (scalar integer)
        nk: Desired dead time (scalar integer)
        tol: tolerance criteria for stopping the algorithm (scalar positive real number)
        u: Input sequence (ndarray)
        y: Output sequence (ndarray)
    Outputs:
        theta: Estimated coefficients of the polynomals F(q^-1) and B(q^-1) (ndarray)
    """
    #Default parameters of the algorithm
    Kmax = 50   #Maximum number of iterations
    L = max(nf, nb+nk)
    #Input testing
    N = len(u)
    #Error handling
    #Algorithm
    cond = True
    k = 1
    Phif = np.zeros((N-L, nf+nb))
    Phi = vec_delay(range(1,nf+1), range(nk, nb+nk), -y, u)
    theta = la.solve(Phi.T.dot(Phi), Phi.T.dot(y[L::]))
    while cond:
        Ahat = np.insert(theta[0:nf], 0, 1)
        for i in range(0, nf + nb):
            Phif[:,i] = sig.lfilter([1], Ahat, Phi[:,i], 0)
        yf = sig.lfilter([1], Ahat, y[L::], 0)
        theta = la.solve(Phif.T.dot(Phif), Phif.T.dot(yf))
        cond = la.norm(Ahat[1::]-theta[0:nf].reshape(1, nf)) > tol and k < Kmax
        k += 1
    # Split theta in vectors a and b
    a = theta[0:nf]
    b = theta[nf+1::]

    return theta

#Avoid a brunch of fors in the methods repeating the same code (this function should only be used as an internal method)
def vec_delay(nu, ny, u, y):
    #Setting the input arguments in a tratable form
    N = len(u)
    u = np.array(u).reshape((N, 1))
    y = np.array(y).reshape((N, 1))
    #Input and error handling
    
    #Algorithm
    u_n = len(nu)
    y_n = len(ny)
    L = max([max(nu),max(ny)])
    Phi = np.zeros((N-L, u_n + y_n))
    for i in range(0, u_n):
        Phi[0:N-L+1, i:i+1] = u[L-nu[i]:N-nu[i]] 
    for i in range(0, y_n):    
        Phi[0:N-L+1, u_n+i:u_n+i+1] = y[L-ny[i]:N-ny[i]]
    return Phi