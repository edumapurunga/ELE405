import sys
sys.path.append('..')
from sysid import vec_delay
import numpy as np

#How to use it:
#
#This function returns a matrix of delayed versions of the vector u and y.
#
#Inputs:
#    nu : an array containing the chosen delays for vector u (numpy array or list)
#    ny : an array containing the chosen delays for vector y (numpy array or list)
#    u  : the data vector (numpy array)
#    y  : the data vector (numpy array)
#    Phi: a matrix of the form: (numpy array)
#Outputs:
#    Phi = 
#Examples:

## vec_delay function test ##
u = np.array(range(0, 10))
y = 3*u
#Here the user can choose which delayed versions (s)he wants for each data vector
phi = vec_delay([1, 2], [0, 1], -y, u)
#In the above example, the user has chosen as regressors for y: y[k-1] and y[k-2] and for u: u[k] and u[k-1] 
## ARX example na = 2, nb = 2, nk = 1
na = 2
nb = 2
nk = 2
phiARX = vec_delay(range(1, na+1), range(nk, nb+nk), -y, u)