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

import sys
sys.path.append('..')
from sysid import ls
import numpy as np
from scipy import signal

N=1000

b=np.array([0, 0.01])
a=np.array([1, -0.99, 0.5])

u1=np.ones(N)
y1=signal.lfilter(b,a,u1)

t = np.arange(N)
u2=np.cos(t/10)
y2=signal.lfilter(b,a,u2)

theta = ls(2,1,u2,y2)

print(theta)