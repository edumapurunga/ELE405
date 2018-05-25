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
from scipy import signal


def ls(na, nb, u, y, nk=0):
    '''

    :param na: number of zeros from A;
    :param nb: number of poles from B;
    :param u: input signal;
    :param y: output signal;
    :param nk: output signal delay;
    :return: coefficients of A and B in this order;
    '''

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
    '''

    :param na: number of zeros from A;
    :param nb: number of poles from B;
    :param u: input signal;
    :param y: output signal;
    :param y2:
    :param nk: output signal delay;
    :return: coefficients of A and B in this order;
    '''

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


def els(na, nb, nc, u, y, nk=0, n=10):
    """
    Implementation of Extended Least Square (ELS) algorithm for
    Auto-Regressive Moving-Averege with eXogenous input(ARMAX) model
    ARMAX model:
                    A*y = B*u + C*e

    :param na: number of zeros from A;
    :param nb: number of poles from B;
    :param nc: number of poles from C;
    :param u: input signal;
    :param y: output signal;
    :param nk: output signal delay;
    :param n: Number of iterations of the recursive algorithm. Default Value is 10;
    :return: coefficients of A,B and C in this order;
    """
    # Number of samples
    samples = np.size(y)

    # Vetor u and y must have same amount of samples
    if samples != np.size(u):
        raise ValueError('Y and U must have same length!')

    # Number of coefficients to be estimated
    # (a_1, a_2, a_3,..., a_na, b_0, b_1, b_2,..., b_nb, c_1, c_2,..., c_nc)
    m = na + nb + 1 + nc

    # Delay maximum needed
    n_max = np.amax([na, nb + nk, nc])

    # In order to estimate the coefficients, we will need to delay the samples.
    # If the maximum order is greater than the number of samples,
    # then it will not be possible!
    if not (samples - n_max > 0):
        raise ValueError('Number of samples should be greater' +
                         'than the maximum order!')

    # Build matrix phi in which will contain y and u shifted in time;
    phi = np.zeros((samples - n_max, m))
    error = np.zeros(samples)
    iteration = n
    # Recursive part
    while iteration >= 0:
        if iteration == n:
            # at the first iteration, we don't have the error yet, thus
            # Least Square is used one time to find the error;
            a, b = ls(na, nb, u, y)
            # initial estimation of c;
            c = a
        else:
            # it points to the first row of phi matrix;
            k = 0
            # Fill phi with y shifted in time from 0 to na
            for i in range(1, na + 1):
                phi[:, k] = -y[n_max - i:samples - i]
                k = k + 1

            # Fill phi with u shifted in time from 0 to nb
            for i in range(nk, nb + nk + 1):
                phi[:, k] = u[n_max - i:samples - i]
                k = k + 1

            # Fill phi with u shifted in time from 0 to nc
            for i in range(1, nc + 1):
                phi[:, k] = error[n_max - i:samples - i]
                k = k + 1

            # If two or more columns of phi are equal,
            # then inverse operation will not be possible!
            if la.matrix_rank(phi) < m:
                raise ValueError('Phi must have rank equal to na+nb+1+nc!')
            # Crop y from n_max to N
            y_aux = y[n_max:samples]
            # Find theta
            R = np.dot(phi.T, phi)
            # If the experiment is not informative:
            if la.matrix_rank(R) < m:
                raise ValueError('Experiment is not informative')
            S = np.dot(phi.T, y_aux)
            theta = la.solve(R, S)
            a = theta[0:na]
            b = theta[na:na + nb + 1]
            c = theta[na + nb + 1: na + nb + 1 + nc]
        iteration = iteration - 1
        # formatting a and c to [1, a] and [1, c]
        a = np.append(np.array([1]), a)
        c = np.append(np.array([1]), c)
        """
        As we already have the values from a b c y and u, we need to compute the error's value.
        we will do as describe in this formula :
                    e = (A/C)*Y - (B/C)*U
        The formula above it's just a rearrangement of the formula presented in the docstring of this function 
        """
        error = signal.lfilter(a, c, y) - signal.lfilter(b, c, u)
    # To return the standard as the ls output;
    a = a[1:]
    c = c[1:]
    return [a, b, c]

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

