import numpy as np
import numpy.linalg as npl
from numpy.linalg import inv
from numpy import random
import scipy as scipy
import scipy.sparse as sparse
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from scipy.sparse import identity
from scipy.sparse import vstack

def max_of_three(triple):
    l,m,r = triple
    if l<m and r<m:
        return True
    return False

def min_of_three(triple):
    l,m,r = triple
    if l>m and r>m:
        return True
    return False

def make_hist(R,no_of_alphas,no_bars):
    Rprof2d=[]

    for i in range(no_of_alphas):
        vals,locs = np.histogram(R[0:1000,i],bins=np.linspace(-1,1,no_bars))
        Rprof2d.append(vals)
    
    return np.array(Rprof2d)

def make_hist_ev(array,no_of_param_iters,period_min,period_max,no_bars):
    Array_ev_2d=[]

    for i in range(no_of_param_iters):
        vals,locs = np.histogram(array[i],bins=np.linspace(period_min,period_max,no_bars))
        Array_ev_2d.append(vals)
    
    return np.array(Array_ev_2d)