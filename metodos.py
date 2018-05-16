#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:59:37 2018

@author: diego
"""


import numpy as np


def arx(u,y):

    N=u.size
    phi=np.c_[u[1:N-1],y[1:N-1]]
    R=phi.T@phi
    S=phi.T@y[2:N]
    theta=np.linalg.solve(R,S)
    
    #print(theta)

    return theta