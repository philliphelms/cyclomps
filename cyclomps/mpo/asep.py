from numpy import ndarray as npndarray
from numpy import ones as npones
from numpy import zeros as npzeros
from numpy import float_
from cyclomps.mpo.ops import *
from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
import collections
from numpy import exp

############################################################################
# General Asymmetric Simple Exclusion Process:
#
#                     _p_
#           ___ ___ _|_ \/_ ___ ___ ___ ___ ___
# alpha--->|   |   |   |   |   |   |   |   |   |---> beta
# gamma<---|___|___|___|___|___|___|___|___|___|<--- delta
#                   /\___|
#                      q
# Functions:
#   return_mpo(N,hamParams):
#       hamParams[0] = alpha  (In at left)
#       hamParams[1] = gamma  (Out at left)
#       hamParams[2] = p      (Forward at site)
#       hamParams[3] = q      (Backward at site)
#       hamParams[4] = beta   (Out at right)
#       hamParams[5] = delta  (In at right)
#       hamParams[6] = s      (bias)
###########################################################################

def return_mpo(N,hamParams,periodic=False):
    if not isinstance(hamParams[0],(collections.Sequence,npndarray)):
        hamParams = val2vecParams(N,hamParams)
    else:
        hamParams = extractParams(N,hamParams)
    if periodic:
        mpo = periodic_mpo(N,hamParams)
    else:
        mpo = open_mpo(N,hamParams)
    mpo = reorder_bonds(mpo)
    return mpo

def open_mpo(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # List to hold all mpos
    mpoL = []
    # Main mpo
    mpo = [None]*N
    for site in range(N):
        # Generic Operator Form
        gen_mpo = array([[I,              z,  z,  z,  z, z],
                            [ep[site-1]*Sm,  z,  z,  z,  z, z],
                            [p[site-1]*v,    z,  z,  z,  z, z],
                            [eq[site]*Sp,    z,  z,  z,  z, z],
                            [q[site]*n,      z,  z,  z,  z, z],
                            [z,             Sp, -n, Sm, -v, I]])
        # Include destruction & creation at site
        gen_mpo[-1,0,:,:] += (ea[site] + ed[site])*Sm -\
                             ( a[site] +  d[site])*v  +\
                             (eb[site] + eg[site])*Sp -\
                             ( b[site] +  g[site])*n
        #gen_mpo[-1,0,:,:] += (a[site] + d[site])*Sm -\
        #                     (a[site] + d[site])*v  +\
        #                     (b[site] + g[site])*Sp -\
        #                     (b[site] + g[site])*n
        # Add operator to mpo
        if (site == 0):
            mpo[site] = expand_dims(gen_mpo[-1,:],0)
        elif (site == N-1):
            mpo[site] = expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_mpo(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_mpo(N,hamParams)
    if p[-1] != 0:
        tmp_op1 = [None]*N
        tmp_op2 = [None]*N
        tmp_op1[-1] = array([[ep[-1]*Sp]])
        tmp_op2[-1] = array([[-p[-1]*n]])
        tmp_op1[0] = array([[Sm]])
        tmp_op2[0] = array([[v]])
        mpoL.append(tmp_op1)
        mpoL.append(tmp_op2)
    if q[0] != 0:
        tmp_op1 = [None]*N
        tmp_op2 = [None]*N
        tmp_op1[-1] = array([[eq[0]*Sm]])
        tmp_op2[-1] = array([[-q[0]*v]])
        tmp_op1[0] = array([[Sp]])
        tmp_op2[0] = array([[n]])
        mpoL.append(tmp_op1)
        mpoL.append(tmp_op2)
    return mpoL

# USEFUL FUNCTIONS ------------------------------------------------------

def exponentiateBias(hamParams):
    (a,g,p,q,b,d,s) = hamParams
    ea = a*exp(s)
    eg = g*exp(-s)
    ep = p*exp(s)
    eq = q*exp(-s)
    eb = b*exp(s)
    ed = d*exp(-s)
    return (ea,eg,ep,eq,eb,ed)

def val2vecParams(N,hamParams):
    # Extract values
    if not isinstance(hamParams[0],(collections.Sequence,npndarray)):
        a = float(hamParams[0])
        aVec = npzeros(N,dtype=float_)
        aVec[0] = a
    else:
        aVec = a
    if not isinstance(hamParams[1],(collections.Sequence,npndarray)):
        g = float(hamParams[1])
        gVec = npzeros(N,dtype=float_)
        gVec[0] = g
    else:
        gVec = g
    if not isinstance(hamParams[2],(collections.Sequence,npndarray)):
        p = float(hamParams[2])
        pVec = p*npones(N,dtype=float_)
    else:
        pVec = p
    if not isinstance(hamParams[3],(collections.Sequence,npndarray)):
        q = float(hamParams[3])
        qVec = q*npones(N,dtype=float_)
    else:
        qVec = q
    if not isinstance(hamParams[4],(collections.Sequence,npndarray)):
        b = float(hamParams[4])
        bVec = npzeros(N,dtype=float_)
        bVec[-1] = b
    else:
        bVec = b
    if not isinstance(hamParams[5],(collections.Sequence,npndarray)):
        d = float(hamParams[5])
        dVec = npzeros(N,dtype=float_)
        dVec[-1] = d
    else:
        dVec = d
    if not isinstance(hamParams[6],(collections.Sequence,npndarray)):
        s = float(hamParams[6])
        sVec = s*npones(N,dtype=float_)
    else:
        sVec = s
    # Convert to vectors
    returnParams = (aVec,gVec,pVec,qVec,bVec,dVec,sVec)
    return returnParams

def extractParams(N,hamParams):
    a = hamparams[0].astype(dtype=float_)
    g = hamparams[1].astype(dtype=float_)
    p = hamparams[2].astype(dtype=float_)
    q = hamparams[3].astype(dtype=float_)
    b = hamparams[4].astype(dtype=float_)
    d = hamparams[5].astype(dtype=float_)
    s = hamparams[6].astype(dtype=float_)
    return (a,g,p,q,b,d,s)

# CURRENT OPERATORS ------------------------------------------------------

def curr_mpo(N,hamParams,periodic=False,singleBond=False,bond=None):
    if not isinstance(hamParams[0],(collections.Sequence,npndarray)):
        hamParams = val2vecParams(N,hamParams)
    else:
        hamParams = extractParams(N,hamParams)
    if singleBond:
        mpo = single_bond_curr(N,hamParams,bond=bond)
    else:
        if periodic:
            mpo = periodic_curr(N,hamParams)
        else:
            mpo = open_curr(N,hamParams)
    mpo = reorder_bonds(mpo)
    return mpo

def single_bond_curr(N,hamParams,bond=None):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # Decide which bond to measure current over
    if bond is None:
        bond = int(N/2)
    # List to hold all mpos
    mpoL = []
    mpo = [None]*N
    # Fill in mpo
    if bond == 'left':
        mpo[0] = array([[(ea[ 0]-ed[ 0])*Sm - (eb[ 0]-eg[ 0])*Sp]])
    elif bond == 'right':
        mpo[-1]= array([[(ea[-1]-ed[-1])*Sm - (eb[-1]-eg[-1])*Sp]])
    else:
        mpo[bond] = array([[Sp,Sm]])
        mpo[bond+1] = array([[ep[bond]*Sm],
                                [-eq[bond+1]*Sp]])
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def open_curr(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # List to hold all mpos
    mpoL = []
    # Main mpo
    mpo = [None]*N
    for site in range(N):
        # Generic Operator Form
        gen_mpo = array([[I,              z,  z, z],
                            [ep[site-1]*Sm,  z,  z, z],
                            [-eq[site]*Sp,    z,  z, z],
                            [z,             Sp, Sm, I]])
        # Include destruction & creation at site
        gen_mpo[-1,0,:,:] += (ea[site] - ed[site])*Sm +\
                             (eb[site] - eg[site])*Sp
        # Add operator to mpo
        if (site == 0):
            mpo[site] = expand_dims(gen_mpo[-1,:],0)
        elif (site == N-1):
            mpo[site] = expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_curr(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_mpo(N,hamParams)
    if p[-1] != 0:
        tmp_op1 = [None]*N
        tmp_op1[-1] = array([[ep[-1]*Sp]])
        tmp_op1[0] = array([[Sm]])
        mpoL.append(tmp_op1)
    if q[0] != 0:
        tmp_op1 = [None]*N
        tmp_op1[-1] = array([[-eq[0]*Sm]])
        tmp_op1[0] = array([[Sp]])
        mpoL.append(tmp_op1)
    return mpoL

# ACTIVITY OPERATORS ------------------------------------------------------

def act_mpo(N,hamParams,periodic=False,singleBond=False,bond=None):
    if not isinstance(hamParams[0],(collections.Sequence,npndarray)):
        hamParams = val2vecParams(N,hamParams)
    else:
        hamParams = extractParams(N,hamParams)
    if singleBond:
        mpo = single_bond_act(N,hamParams,bond=bond)
    else:
        if periodic:
            mpo = periodic_act(N,hamParams)
        else:
            mpo = open_act(N,hamParams)
    mpo = reorder_bonds(mpo)
    return mpo

def single_bond_act(N,hamParams,bond=None):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # Decide which bond to measure current over
    if bond is None:
        bond = int(N/2)
    # List to hold corresponding mpos
    mpoL = []
    mpo = [None]*N
    # Fill in mpo
    if bond == 'left':
        mpo[0]  = array([[(ea[ 0]+ed[ 0])*Sm + (eb[ 0]+eg[ 0])*Sp]])
    elif bond == 'right':
        mpo[-1] = array([[(ea[-1]+ed[-1])*Sm + (eb[-1]+eg[-1])*Sp]])
    else:
        mpo[bond] = array([[Sp,Sm]])
        mpo[bond+1] = array([[ep[bond]*Sm],
                                [eq[bond+1]*Sp]])
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def open_act(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # List to hold all mpos
    mpoL = []
    # Main mpo
    mpo = [None]*N
    for site in range(N):
        # Generic Operator Form
        gen_mpo = array([[I,              z,  z, z],
                            [ep[site-1]*Sm,  z,  z, z],
                            [eq[site]*Sp,    z,  z, z],
                            [z,             Sp, Sm, I]])
        # Include destruction & creation at site
        gen_mpo[-1,0,:,:] += (ea[site] + ed[site])*Sm +\
                             (eb[site] + eg[site])*Sp
        # Add operator to mpo
        if (site == 0):
            mpo[site] = expand_dims(gen_mpo[-1,:],0)
        elif (site == N-1):
            mpo[site] = expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_act(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_mpo(N,hamParams)
    if p[-1] != 0:
        tmp_op1 = [None]*N
        tmp_op1[-1] = array([[ep[-1]*Sp]])
        tmp_op1[0] = array([[Sm]])
        mpoL.append(tmp_op1)
    if q[0] != 0:
        tmp_op1 = [None]*N
        tmp_op1[-1] = array([[eq[0]*Sm]])
        tmp_op1[0] = array([[Sp]])
        mpoL.append(tmp_op1)
    return mpoL
