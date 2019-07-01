"""
Linear Algebra Tools

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""
from cyclomps.tools.params import *# import params
import numpy as np
from scipy.linalg import eig as sleig
from scipy.linalg import orth as slorth


# -------------------------------------------------
# Tensor Allocation
array      = np.array
eye        = np.eye
ones       = np.ones
rand       = np.random.random
zeros      = np.zeros
zeros_like = np.zeros_like

def rand(dim,dtype=None):
    return np.random.random(dim).astype(dtype)

def save_ten(ten,fname):
    np.save(fname,ten)

def load_ten(dim,fname,dtype=None):    
    return np.load(fname+'.npy')

def to_nparray(a):
    return a

def from_nparray(a):
    return a

# -------------------------------------------------
# Linear Algebra
abss       = np.abs
dot        = np.dot
einsum     = np.einsum
qr         = np.linalg.qr
summ       = np.sum
prod       = np.prod
sqrt       = np.sqrt
log2       = np.log2
orth       = slorth
argsort    = np.argsort
take       = np.take
def eig(h):
    E,vecs = sleig(h)
    inds = np.argsort(E)[::-1]
    E = E[inds]
    vecs = vecs[:,inds]
    return E,vecs
def svd(ten):
    return np.linalg.svd(ten,full_matrices=False)

# -------------------------------------------------
# Tensor Manipulation
conj       = np.conj
diag       = np.diag
diagonal   = np.diagonal
imag       = np.imag
ravel      = np.ravel
real       = np.real
reshape    = np.reshape
transpose  = np.transpose
expand_dims= np.expand_dims
