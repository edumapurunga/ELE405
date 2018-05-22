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

def ls(na, nb, u, y, nk=0):

    # Number of samples
    N = np.size(y);

    # Vetor u and y must have same amount of samples
    if N != np.size(u):
        raise ValueError('Y and U must have same length!');

    # Number of coefficients to be estimated
    # (a_1, a_2, a_3,..., a_na, b_0, b_1, b_2, b_nb)
    M = na+nb+1;

    # Delay maximum needed
    n_max = np.amax([na, nb+nk]);

    # In order to estimate the coeffiecients, we will need to delay the samples.
    # If the maximum order is greater than the number of samples,
    # then it will not be possible!
    if not (N - n_max > 0):
        raise ValueError('Number of samples should be greater' &
                         'than the maximum order!');

    # Build matrix phi in which will contain y and u shifted in time
    phi = np.zeros((N-n_max, M));

    k = 0;

    # Fill phi with y shifted in time from 0 to nb
    for i in range(1,na+1):
        phi[:, k] = y[n_max-i:N-i];
        k=k+1;

    # Fill phi with u shifted in time from 0 to nb
    for i in range(nk,nb+nk+1):
        phi[:, k] = u[n_max-i:N-i];
        k=k+1;

    # If two or more columns of phi are equal,
    # then inverse operation will not be possible!
    if (la.matrix_rank(phi) < M):
        raise ValueError('Phi must have rank equal to na+nb+1!');

    # Crop y from n_max to N
    y = y[n_max:N];

    # Find theta
    R=np.dot(phi.T,phi);
    S=np.dot(phi.T,y);
    theta=la.solve(R,S);

    # Split theta in vectors a and b
    a = theta[0:na];
    b = theta[na:na+nb+1];

    return [a, b];

def iv(na, nb, u, y, u2, y2, nk=0):

    # Number of samples
    N = np.size(y)

    # Vetors u, y, u2 and y2 must have same amount of samples
    if (N != np.size(u)) or (N != np.size(u2)) or (N != np.size(y2)):
        raise ValueError('Y, Y2, U and U2 must have same length!')

    # Number of coefficients to be estimated
    # (a_1, a_2, a_3,..., a_na, b_0, b_1, b_2, b_nb)
    M = na + nb + 1

    # Delay maximum needed
    n_max = np.amax([na, nb+nk])

    # In order to estimate the coeffiecients, we will need to delay the samples.
    # If the maximum order is greater than the number of samples,
    # then it will not be possible!
    if not (N - n_max > 0):
        raise ValueError('Number of samples should be greater' &
                         'than the maximum order!')

    # Build matrix phi in which will contain y and u shifted in time
    phi = np.zeros((N-n_max, M))

    # Build matrix csi in which will contain y2 and u2 shifted in time
    csi = np.zeros((N-n_max, M))

    k = 0

    # Fill phi/csi with y/y2 shifted in time from 0 to nb
    for i in range(1, na+1):
        phi[:, k] = y[n_max-i:N-i]
        csi[:, k] = y2[n_max-i:N-i]
        k=k+1

    # Fill phi/csi with u/u2 shifted in time from 0 to nb
    for i in range(nk,nb+nk+1):
        phi[:, k] = u[n_max-i:N-i]
        csi[:, k] = u2[n_max-i:N-i]
        k=k+1

    # If two or more columns of phi are equal,
    # then inverse operation will not be possible!
    if (la.matrix_rank(phi) < M):
        raise ValueError('Phi must have rank equal to na+nb+1!')

    # Crop y from n_max to N
    y = y[n_max:N]

    # Find theta
    R = np.dot(csi.T, phi)
    S = np.dot(csi.T, y)
    theta = la.solve(R, S)

    # Split theta in vectors a and b
    a = theta[0:na]
    b = theta[na:na+nb+1]

    return [a, b]
