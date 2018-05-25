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

def ls(na, nb, nk, u, y):

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
    phi = vec_delay(range(1, na+1), -y, range(nk, nk+nb+1), u)

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
    N = np.size(y) #Same of LS here @edumapurunga

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
    #Input testing
    #Error handling
    #Algorithm
    cond = True
    k = 1
    #First Estimate using the Least Squares method
    Aold, Bold = ls(nf, nb, nk, u, y)
    while cond:
        #Filter for whitening the noise term
        Ahat = np.insert(Aold, 0, 1)
        #Filtered input
        uf = sig.lfilter([1], Ahat, u, 0)
        #Filtered output
        yf = sig.lfilter([1], Ahat, y, 0)
        #Solve a LS problem with the new filtered data
        Anew, Bnew = ls(nf, nb, nk, uf, yf)
        #Stopping Criteria: 1) Little variation between the norm of the old and new vector. 2) Maximum iterations reached
        cond = la.norm(Ahat[1::]-Anew.reshape(1, nf)) > tol and k < Kmax
        #Update
        Aold = Anew
        k += 1
    return [Anew, Bnew]

def kalman(A, B, C, D, G, Q, R, u, y, xo, Po):
    """
    This function implements the Kalman filter that estimates the states of a 
    system defined as:
        
        x[k+1] = A[k]x[k] + B[k]u[k] + G[k]w[k] (State Equation)
        y[k]   = C[k]x[k] + D[k]u[k]       v[k] (Measurement Equation)
        
    with known inputs u, outputs y and the covariances matrixs Q[k] = E[w[k]w[j]']
    and R[k] = E[v[k]v[j]'].
    Inputs:
        A - the
    """
    #Input testing
    Nu, r = u.shape
    Ny, p = y.shape
    n = xo.shape
    n = n[0]
    N = Nu
    #Error handling
    #Data Storage
    xkk = np.zeros((n, N)) #Measurement update
    xkk1= np.zeros((n, N)) #Time update
    Kk  = np.zeros((n, N)) #Kalman gain
    Ekk = np.zeros((N, 1)) #Sum of the squared estimation error for x[k|k]
    Vkk = np.zeros((n, N)) #Variance of each state for x[k|k]
    Ekk1= np.zeros((N, 1)) #Sum of the squared estimation error for x[k|k-1]
    Vkk1= np.zeros((n, N)) #Variance of each state for x[k|k-1]
    #Discete case
    #Initial conditions
    P = Po
    x = xo
    for i in range(0, N):
        #Save the delayed prediction of Kalman Filter
        xkk1[:,i:i+1] = x
        Ekk1[i] = np.trace(P)
        Vkk1[:,i] = np.diag(P)
        #Measurement update
        M = np.dot(P.dot(C.T), la.pinv(C.dot(P.dot(C.T)) + R))
        x = x + M.dot(y[i]-(C.dot(x)+D.dot(u[i]))) #x[k|k]
        P = np.dot(np.eye(n)-M.dot(C), P)          #P[k|k]
        #Save the Data
        xkk[:,i:i+1] = x
        Ekk[i] = np.trace(P)
        Vkk[:,i] = np.diag(P)
        #Time update
        x = A.dot(x) + B.dot(u[i:i+1]) #x[k+1|k]
        P = np.dot(A.dot(P), A.T) + np.dot(G.dot(Q), G.T)
        #Kalman Filter
        Kk[:,i:i+1] = A.dot(M)
        
    return (Kk, xkk, Vkk, xkk1, Vkk1)

#################################################################################################
#Auxiliary functions
def vec_delay(nu, u, ny = [], y = [], nw = [], w = []):
    """
    This function returns a matrix of delayed versions of the vector u and y.

    Inputs:
        nu : an array containing the chosen delays for vector u (numpy array or list)
        ny : an array containing the chosen delays for vector y (numpy array or list)
        nw : an array containing the chosen delays for vector w (numpy array or list)
        u  : the data vector (numpy array)
        y  : the data vector (numpy array)
        w  : the data vector (numpy array)
    Outputs:
        Phi: a matrix of the form: (numpy array)
        Phi = 
    """    
    #Number of arguments sent by the user
    N = len(u)
    u_n = len(nu)
    D = nu
    u = np.array(u).reshape((N, 1))
    Dd = u
    narg = 1
    y_n = 0
    maxny = 0
    w_n = 0
    maxnw = 0
    if len(y):
        narg += 1
        y = np.array(y).reshape((N, 1))
        D = [D, ny]
        Dd = [Dd, y]
        maxny = max(ny)
        y_n = len(ny)
    if len(w):
        narg += 1
        w = np.array(w).reshape((N, 1))
        D = [D, nw]
        Dd = [Dd, w]
        maxnw = max(nw)
        w_n = len(ny)
    #Input and error handling
    
    #Algorithm
    L = max([max(nu), maxny, maxnw])
    Phi = np.zeros((N-L, u_n+y_n+w_n))
    cd = 0
    ci = 0
    for d in D:
        for i in range(0, len(d)):
            Phi[0:N-L+1, ci:ci+1] = Dd[cd][L-d[i]:N-d[i]]
            ci +=1
        cd += 1
    return Phi