"""
Tools for Matrix Product Operators

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""

from cyclomps.tools.utils import *
from numpy import eye as npeye
from numpy import array as nparray
from numpy import int as npint
from scipy.linalg import expm as sla_expm
import copy

def expm(m,a=1.):
    """
    Take the exponential of a matrix
    """
    m = to_nparray(m)
    m *= a
    m = sla_expm(m)
    m = from_nparray(m)
    return m

def quick_op(op1,op2):
    """
    Convert two local operators (2 legs each)
    into an operator between sites (4 legs)
    """
    return einsum('io,IO->iIoO',op1,op2)

def list_configs(d):
    """
    List all of the possible configurations
    """
    if len(d) == 1:
        configs = []
        for config in range(d[0]):
            configs.append([config])
        return configs
    else:
        old_configs = list_configs(d[1:])
        new_configs = []
        for config in range(d[0]):
            for old_config_ind in range(len(old_configs)):
                new_configs.append( [config] + old_configs[old_config_ind] )
        return new_configs

def mpo2mat(mpoList,dloc=2):
    """
    Convert an mpoList to a matrix operator

    Args:
        mpoList : 1D Array
            An array that contains a list of mpos, each of which
            is a list with an array for each site in the lattice
            representing the mpo at that site.

    Returns:
        mat : 2D Array
            A 2D array containing the matrix version of the input
            mpo
    """
    mpiprint(5,'Converting mpoList to a matrix operator')

    # Get details about mpo
    nOps = len(mpoList)
    nSite= len(mpoList[0])

    # Figure out local bond dimensions
    d = [None]*nSite
    for site in range(nSite):
        for op in range(nOps):
            if mpoList[op][site] is not None:
                _,dloc,_,_ = mpoList[op][site].shape
        d[site] = dloc

    # Allocate memory for matrix operator
    matDim = prod(d)
    mat = zeros((matDim,matDim))

    configs = list_configs(d)
    # Loop through all possible incoming states
    for i in range(len(configs)):
        # Figure out actual configuration
        iconfig = configs[i]
        # Loop through all possible outgoing states
        for j in range(prod(d)):
            # Figure out actual configuration
            jconfig = configs[j]
            # Loop through all operators
            for opInd in range(nOps):
                tmp_mat = ones((1,1))
                # Loop through sites
                for site in range(nSite):
                    istate = iconfig[site]
                    jstate = jconfig[site]
                    if mpoList[opInd][site] is not None:
                        tmp_mat = einsum('ab,bc->ac',tmp_mat,mpoList[opInd][site][:,istate,jstate,:])
                    else:
                        # Introduce identity operator (PH - Lazy way, fix this)
                        identity = array(nparray([[npeye(2)]]))
                        identity = einsum('abpq->apqb',identity)
                        tmp_mat = einsum('ab,bc->ac',tmp_mat,identity[:,istate,jstate,:])
                print(tmp_mat[0,0])
                mat[i,j] += tmp_mat[0,0]
    return mat

def mpo_conj_trans(mpoList,copy=True):
    """
    Return the conjugate transpose of the input mpo

    Args:
        mpoList : 1D Array
            An array that contains a list of mpos, each of which
            is a list with an array for each site in the lattice
            representing the mpo at that site.

    Returns:
        mpoList : 1D Array
            A conjugated, transposed version of the input mpo
    """
    mpiprint(5,'Taking the conjugate transpose of the mpo list')

    # Copy the mpo list (if desired)
    if copy: mpoList = copy_mpo(mpoList)

    # Get details about mpo
    nOps = len(mpoList)
    nSite= len(mpoList[0])

    # Sweep through operators
    for op in range(nOps):
        # Sweep through sites
        for site in range(nSite):
            if mpoList[op][site] is not None:
                # Conjugate transpose of mpo at site
                mpoList[op][site] = einsum('apqb->aqpb',mpoList[op][site]).conj()
    return mpoList


def copy_mpo(mpoList):
    """
    Copy an mpo list

    Args:
        mpoList : 1D Array
            An array that contains a list of mpos, each of which
            is a list with an array for each site in the lattice
            representing the mpo at that site.

    Returns:
        mpoList : 1D Array
            A copy of the input mpo list
    """
    mpiprint(7,'Copying mpo list')

    # Get details about mpo
    nOps = len(mpoList)
    nSite = len(mpoList[0])

    # Create list to copy mpo list into
    mpoListCopy = []

    # Loop over operators
    for op in range(nOps):
        mpo = [None]*nSite
        # Loop through sites
        for site in range(nSite):
            if mpoList[op][site] is not None:
                mpo[site] = copy.deepcopy(mpoList[op][site])
        # Add new mpo to copied mpoList
        mpoListCopy.append(mpo)

    return mpoListCopy

def reorder_bonds(mpoList):
    """
    If an mpo has been created using

    op = array([[I, z],
                [z, I]])

    then the order of the bonds is:
    [auxilliary, auxilliary, physical, physical]

    this function sweeps through all operators
    in an mpo and reorders the bond to:
    [auxilliary, physical, physical, auxilliary]

    Args:
        mpoList : 1D Array
            An array that contains a list of mpos, each of which
            is a list with an array for each site in the lattice
            representing the mpo at that site.

    Returns:
        mpoList : 1D Array
            A copy of the input mpo list
    """
    mpiprint(8,'Reordering mpo bonds')

    # Get relevant info
    nOps = len(mpoList)
    nSite = len(mpoList[0])

    # Loop through all operators
    for op in range(nOps):
        # Loop through all sites
        for site in range(nSite):
            # Check if operator is a tensor or None
            if mpoList[op][site] is not None:
                mpoList[op][site] = einsum('abpq->apqb',mpoList[op][site])

    # Return reordered bonds mpoList
    return mpoList

def mpo_local_dim(mpoList):
    """
    Determine the local bond dimensions from an mpoList

    Args:
        mpoList : 1D array
            An array that contains a list of mpos, each of which
            is a list with an array for each site in the lattice
            representing the mpo at that site.

    Returns:
        d : 1D array
            A list corresponding to the local state space dimension
            of each site, as determined by the mpo
    """
    mpiprint(7,'Determining local bond dimensions')

    # Get info from mpo list
    nOps = len(mpoList)
    nSite = len(mpoList[0])

    # Create list of local bond dims
    d = [0]*nSite

    # Loop through all sites
    for site in range(nSite):
        d[site] = 1
        # Loop through all operators, since some are None
        for op in range(nOps):
            if mpoList[op][site] is not None:
                # If not None, then use this dimension
                _,dtmp,_,_ = mpoList[op][site].shape
                d[site] = dtmp

    # Return Results
    return d
