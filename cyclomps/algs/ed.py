"""
Extremely naive exact diagonalization scheme

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""
from cyclomps.tools.mpo_tools import *
import scipy.linalg as sla
from numpy import argsort as npargsort
from numpy import real as npreal

def ed(mpo,left=False):
    # Convert mpo into a matrix
    H = mpo2mat(mpo)
    H = to_nparray(H)
    (nx,ny) = H.shape
    # Solve eigenproblem
    if left:
        e,vl,vr = sla.eig(H,left=True)
        inds = npargsort(npreal(e))[::-1]
        e = e[inds]
        vl = vl[:,inds]
        vr = vr[:,inds]
        e = from_nparray(e)
        vl = from_nparray(vl)
        vr = from_nparray(vr)
        return e,vl,vr
    else:
        e,vr = sla.eig(H,left=False)
        inds = npargsort(npreal(e))[::-1]
        e = e[inds]
        vr = vr[:,inds]
        e = from_nparray(e)
        vr = from_nparray(vr)
        return e,vr
