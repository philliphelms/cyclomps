"""
Create the mpo for the totally asymmetric simple exclusion process

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019


 Totally Asymmetric Simple Exclusion Process description:

                     _1_
           ___ ___ _|_ \/_ ___ ___ ___ ___ ___
 alpha--->|   |   |   |   |   |   |   |   |   |---> beta
          |___|___|___|___|___|___|___|___|___|


"""

from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import *
from cyclomps.mpo.ops import *
from numpy import exp

def return_mpo(N,hamParams,periodic=False):
    if periodic:
        mpo = periodic_mpo(N,hamParams)
    else:
        mpo = open_mpo(N,hamParams)
    mpo = reorder_bonds(mpo)
    return mpo

def open_mpo(N,hamParams):
    # Unpack Ham Params ############################
    a = hamParams[0]
    b = hamParams[1]
    s = hamParams[2]
    exp_a = a*exp(s)
    exp_b = b*exp(s)
    exp_p = 1.*exp(s)
    # Create MPO ###################################
    # List to hold all Operators (only one here though)
    mpoL = []
    # Single operator
    mpo = [None]*N
    mpo[0] = array([[exp_a*Sm-a*v, exp_p*Sp, -n, I]])
    for site in range(1,N-1):
        mpo[site] = array([[I,  z,         z, z],
                           [Sm, z,         z, z],
                           [v,  z,         z, z],
                           [z,  exp_p*Sp, -n, I]])
    mpo[-1] = array([[I],
                    [Sm],
                    [v],
                    [(exp_b*Sp-b*n)]])
    # Add single mpo to list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_mpo(N,hamParams):
    # Unpack Ham Params ############################
    a = hamParams[0]
    b = hamParams[1]
    s = hamParams[2]
    exp_a = a*exp(s)
    exp_b = b*exp(s)
    exp_p = 1.*exp(s)
    # Create MPO ###################################
    # List to hold all Operators (only one here though)
    mpoL = []
    # Main operator
    mpo = [None]*N
    # General operator form:
    gen_mpo = array([[I,  z,         z, z],
                        [Sm, z,         z, z],
                        [v,  z,         z, z],
                        [z,  exp_p*Sp, -n, I]])
    mpo[0] = expand_dims(gen_mpo[-1,:],0)
    for site in range(1,N-1):
        mpo[i] = array(gen_mpo)
    mpo[-1] = expand_dims(gen_mpo[:,0],1)
    mpoL.append(mpo)
    # Periodic terms:
    mpo_p1 = [None]*N
    mpo_p2 = [None]*N
    mpo_p1[-1] = array([[exp_p*Sp]])
    mpo_p2[-1] = array([[-n]])
    mpo_p1[0] = array([[Sm]])
    mpo_p2[0] = array([[v]])
    mpoL.append(mpo_p1)
    mpoL.append(mpo_p2)
    return mpoL

def curr_mpo(N,hamParams,periodic=False):
    if periodic:
        curr = periodic_curr(N,hamParams)
    else:
        curr = open_curr(N,hamParams)
    curr = reorder_bonds(curr)
    return curr

def open_curr(N,hamParams):
    # Unpack Ham Params ############################
    a = hamParams[0]
    b = hamParams[1]
    s = hamParams[2]
    exp_a = a*exp(s)
    exp_b = b*exp(s)
    exp_p = 1.*exp(s)
    # Create MPO ###################################
    # List to hold all Operators (only one here though)
    mpoL = []
    # Single operator
    mpo = [None]*N
    mpo[0] = array([[exp_a*Sm, exp_p*Sp, I]])
    for site in range(1,N-1):
        mpo[site] = array([[I,  z,        z],
                           [Sm, z,        z],
                           [z,  exp_p*Sp, I]])
    mpo[-1] = array([[I],
                    [Sm],
                    [exp_b*Sp]])
    # Add single mpo to list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_curr(N,hamParams):
    # Unpack Ham Params ############################
    a = hamParams[0]
    b = hamParams[1]
    s = hamParams[2]
    exp_a = a*exp(s)
    exp_b = b*exp(s)
    exp_p = 1.*exp(s)
    # Create MPO ###################################
    # List to hold all Operators (only one here though)
    mpoL = []
    # Main operator
    mpo = [None]*N
    # General operator form:
    gen_mpo = array([[I,  z,       z],
                        [Sm, z,       z],
                        [z,  exp_p*Sp,I]])
    mpo[0] = expand_dims(gen_mpo[-1,:],0)
    for site in range(1,N-1):
        mpo[i] = array(gen_mpo)
    mpo[-1] = expand_dims(gen_mpo[:,0],1)
    mpoL.append(mpo)
    # Periodic terms:
    mpo_p1 = [None]*N
    mpo_p1[-1] = array([[exp_p*Sp]])
    mpo_p1[0] = array([[Sm]])
    mpoL.append(mpo_p1)
    return mpoL

# TEBD Operators --------------------------
def return_tebd_ops(N,hamParams):
    # Unpack Ham Params ############################
    p = 1.
    a = hamParams[0]
    b = hamParams[1]
    s = hamParams[2]
    exp_a = a*exp(s)
    exp_b = b*exp(s)
    exp_p = p*exp(s)
    # Create Operators ###################################
    # List to hold all operators
    op_list = []
    # Initial Site
    op  = exp_a *quick_op(Sm, I)
    op -= a     *quick_op( v, I)
    op += exp_p *quick_op(Sp,Sm)
    op -= p     *quick_op( n, v)
    op_list.append(op)
    # Central Sites
    for site in range(1,N-2):
        op  = exp_p *quick_op(Sp,Sm)
        op -= p     *quick_op( n, v)
        op_list.append(op)
    # Last Site
    op  = exp_p * quick_op(Sp,Sm)
    op -= p     * quick_op( n, v)
    op += exp_b * quick_op( I,Sp)
    op -= b     * quick_op( I, n)
    op_list.append(op)

    return op_list
