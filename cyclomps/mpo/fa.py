from numpy import float_
from cyclomps.mpo.ops import *
from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
import collections
from numpy import exp

############################################################################
# FA Model
#
# Hamiltonian: 
#   W = \sum_i (n_{i-1}+n_{i+1})[ e^{-\lambda} ( c\sigma_i^+ + (1-c)\sigma_i^-)
#                                               -c(1-n_i)    - (1-c)n_i        ]
#
# Functions:
#   return_mpo(N,hamParams):
#       hamParams[0] = c      (flip rate)
#       hamParams[1] = s      (bias)
###########################################################################

def return_mpo(N,hamParams,periodic=False,hermitian=False):
    if not isinstance(hamParams[0],(collections.Sequence)):
        hamParams = val2vecParams(N,hamParams)
    else:
        hamParams = extractParams(N,hamParams)
    if periodic:
        mpo = periodic_mpo(N,hamParams,hermitian=hermitian)
    else:
        mpo = open_mpo(N,hamParams,hermitian=hermitian)
    mpo = reorder_bonds(mpo)
    return mpo

def open_mpo(N,hamParams,hermitian=False):
    # Extract parameter values
    (c,s,mu) = hamParams
    # List to hold all mpos
    mpoL = []
    # Main mpo
    mpo = [None]*N
    for site in range(N):
        # Generic Operator Form
        ci = c[site]
        es = exp(-s[site])
        mui= mu[site]
        if hermitian:
            flipOp = (es*sqrt(ci*(1.-ci))*X  - ci*v - (1.-ci)*n)
        else:
            flipOp = ci*es*Sp + (1-ci)*es*Sm - ci*v - (1.-ci)*n
        # Set up mpo operators
        if (site == 0):
            mpo[site] = array([[z, n, flipOp, I]])
        elif (site == N-1):
            mpo[site] = array([[I],[flipOp],[n],[z]])
        else:
            mpo[site] = array([[I,       z,      z, z],
                               [flipOp,  z,      z, z],
                               [n,       z,      z, z],
                               [mui*n,   n, flipOp, I]])
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_mpo(N,hamParams):
    # Extract parameter values
    (c,s) = hamParams
    # Get main mpo from open_mpo
    mpoL = open_mpo(N,hamParams)
    # Add in Periodic Terms
    perOp1 = [None]*N
    perOp2 = [None]*N
    site = 0
    if hermitian:
        flipOp = es*sqrt(ci*(1.-ci))*X-ci*v-(1.-ci)*n
    else:
        flipOp = ci*es*Sp+(1-ci)*es*Sm-ci*v-(1.-ci)*n
    perOp1[0]  = array([[flipOp]])
    perOp1[-1] = array([[n]])
    site = -1
    if hermitian:
        flipOp = es*sqrt(ci*(1.-ci))*X-ci*(1.-n)-(1.-ci)*n
    else:
        flipOp = ci*es*Sp+(1-ci)*es*Sm-ci*(1.-n)-(1.-ci)*n
    perOp2[0]  = array([[n]])
    perOp2[-1] = array([[flipOp]])
    # Add to mpo list
    mpoL.append(perOp1)
    mpoL.append(perOp2)
    return mpoL

# USEFUL FUNCTIONS ------------------------------------------------------

def val2vecParams(N,hamParams):
    # Extract values
    if not isinstance(hamParams[0],(collections.Sequence)):
        c = float(hamParams[0])
        cVec = c*ones(N,dtype=float_)
        cVec[0] = c
    else:
        cVec = c
    if not isinstance(hamParams[1],(collections.Sequence)):
        s = float(hamParams[1])
        sVec = s*ones(N,dtype=float_)
    else:
        sVec = s
    if not isinstance(hamParams[2],(collections.Sequence)):
        mu = float(hamParams[2])
        muVec = mu*ones(N,dtype=float_)
    else:
        muVec = mu
    # Convert to vectors
    returnParams = (cVec,sVec,muVec)
    return returnParams

def extractParams(N,hamParams):
    c = hamparams[0].astype(dtype=float_)
    s = hamparams[1].astype(dtype=float_)
    mu= hamparams[2].astype(dtype=float_)
    return (c,s)

# ACTIVITY OPERATORS ------------------------------------------------------
# Not implemented yet...
