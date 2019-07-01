"""
Linear Algebra Tools

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""
from cyclomps.tools.params import *# import params
import ctf 
from numpy import array as nparray
from numpy import expand_dims as npexpand_dims
from numpy import prod as npprod
from numpy import argsort as npargsort
from numpy import sqrt as npsqrt
from numpy import log2 as nplog2
from numpy import complex128
from numpy import complex64
from numpy import complex_
from numpy import float_
from scipy.linalg import eig as sleig
from scipy.linalg import orth as slorth

# Tensor Allocation
def array(tens,dtype=None,copy=True,subok=False,ndimin=0):
    ten = nparray(tens)
    return ctf.from_nparray(ten)
eye        = ctf.eye
ones       = ctf.ones
rand       = ctf.random.random
def rand(shape,p=None,dtype=None):
    if dtype == complex128 or dtype == complex64:
        ten_real = ctf.zeros(shape,dtype=float_)
        ten_real.fill_random()
        ten = ctf.zeros(shape,dtype=complex_)
        ten += ten_real
    else:
        ten = ctf.zeros(shape,dtype=dtype,sp=False)
        ten.fill_random()
    return ten
zeros      = ctf.zeros
zeros_like = ctf.zeros_like

# Linear Algebra
abss       = ctf.abs
dot        = ctf.dot
einsum     = ctf.einsum
qr         = ctf.qr
summ       = ctf.sum
svd        = ctf.svd
take       = ctf.take
def argsort(vals):
    vals = to_nparray(vals)
    inds = npargsort(vals)
    return inds.tolist()
def log2(a):
    try:
        res = ctf.from_nparray(nplog2(ctf.to_nparray(a)))
    except:
        res = 0.
    return res
def prod(a):
    return npprod(ctf.to_nparray(a))
def sqrt(a):
    a = ctf.to_nparray(a)
    a = npsqrt(a)
    a = ctf.from_nparray(a)
    return a
def eig(H):
    H = to_nparray(H)
    E,vecs = sleig(H)
    inds = npargsort(E)[::-1]
    E = E[inds]
    vecs = vecs[:,inds]
    vecs = from_nparray(vecs)
    return E,vecs
def orth(v):
    v = to_nparray(v)
    v = slorth(v)
    v = from_nparray(v)
    return v


# Tensor Manipulation
conj       = ctf.conj
diag       = ctf.diag
diagonal   = ctf.diagonal
imag       = ctf.imag
ravel      = ctf.ravel
real       = ctf.real
reshape    = ctf.reshape
transpose  = ctf.transpose
to_nparray = ctf.to_nparray
from_nparray=ctf.from_nparray
def expand_dims(ten,ax):
    ten = ctf.to_nparray(ten)
    ten = npexpand_dims(ten,ax)
    return from_nparray(ten)
# Save & Load Tensors
def save_ten(ten,fname):
    ten.write_to_file(fname)
def load_ten(dim,fname,dtype=None,):
    ten_real = ctf.zeros(dim,dtype=float_)
    ten_real.fill_random()
    ten = ctf.zeros(dim,dtype=dtype)
    ten = ten_real + 0.j
    ten.read_from_file(fname)
    return ten
# Overwrite functions that are different for sparse tensors
if USE_SPARSE:
    diag = ctf.spdiag
    eye = ctf.speye
    def rand(shape,p=None,dtype=None):
        if dtype == complex128 or dtype == complex64:
            ten_real = ctf.zeros(shape,dtype=float_,sp=True)
            ten_real.fill_sp_random()
            ten = ctf.zeros(shape,dtype=dtype,sp=True)
            ten = ten_real+0.j
        else:
            ten = ctf.zeros(shape,dtype=dtype,sp=True)
            ten_real.fill_sp_random()
        return ten
    def zeros(shape,dtype=None):
        return ctf.zeros(shape,dtype=dtype,sp=True)
    def load_ten(dim,fname,dtype=None,sp=True):
        ten = zeros(dim,dtype=dtype,sp=sp)
        ctf.svd(ten)
        ten.read_from_file(fname)
        return ten
