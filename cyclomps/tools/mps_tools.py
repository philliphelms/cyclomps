"""
Tools for Matrix Product States

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

.. To Do:
    Increase Maximum Bond Dimension
    Save and Load full MPS Lists
    Compute probabilities of all configurations
    Move gauge between arbitrary sites

"""

from cyclomps.tools.utils import *
from cyclomps.tools.params import *
from shutil import copyfile
import re
import os
from numpy import float_,complex_,int

def create_mps_list(d,mbd,nStates,
                    sparse=False,dtype=float_,
                    periodic=False,fixed_bd=False,
                    subdir='mps'):
    """
    Create a list containing multiple MPS
    
    Args:
        d : 1D Array
            A list of the local state space dimension
            for the system.
        mbd : int
            Maximum Retained Bond Dimension
        nStates : int
            The number of MPS contained in the list
            corresponding to the number of desired states
        
    Kwargs:
        sparse : bool
            If true, the initialized mps will be sparse
        dtype :
            Indicate the data type for the tensors
        periodic : bool
            If this is true, then create a list of periodic
            mpss, meaning that left and right tensors have 
            dimension [d x mbd x mbd]. If this is False,
            these have dimension [d x 1 x mbd] and 
            [d x mbd x 1]. 
        fixed_bd : bool
            If using open boundary conditions (i.e. periodic==False)
            this ensures that all bond dimensions are constant
            throughout the MPS, i.e. mps[0].dim = (1 x d[0] x mbd)
            instead of mps[0].dim = (1 x d[0] x d[0]), and so forth.

    Returns
    """
    # Create required subdirectory
    if not (subdir[0] == '/'): subdir = '/'+subdir
    subdir_ind = 0
    created = False
    while not created:
        if not os.path.exists(CALCDIR+subdir+str(subdir_ind)):
            os.mkdir(CALCDIR+subdir+str(subdir_ind))
            subdir = subdir + str(subdir_ind)
            created = True
        subdir_ind += 1

    # Loop through and create all MPS
    mpsList = []
    for state in range(nStates):
        mpsList.append(create_rand_mps(d,mbd,state,
                                       sparse=sparse,dtype=dtype,
                                       periodic=periodic,fixed_bd=fixed_bd,
                                       loc=CALCDIR+subdir))
    return mpsList

def create_rand_mps(d,mbd,state,
                    sparse=False,dtype=float_,
                    periodic=False,fixed_bd=False,
                    loc='./'):
    """
    Create a list containing a matrix product state.

    Args:
        d : 1D Array
            A list of the local state space dimension
            for the system
        mbd : int
            Maximum Retained Bond dimension
        state : int
            Specifies whether the MPS is for the ground state (0),
            1st excited (1), 2nd excited (2), etc. 
            ..note Specifies directory in which the mps is saved

    Kwargs:
        sparse : bool
            If true, the initialized mps will be sparse
        dtype :
            Indicate the data type for the tensors
        periodic : bool
            If this is true, then create a list of periodic
            mpss, meaning that left and right tensors have 
            dimension [d x mbd x mbd]. If this is False,
            these have dimension [d x 1 x mbd] and 
            [d x mbd x 1]. 
        fixed_bd : bool
            If using open boundary conditions (i.e. periodic==False)
            this ensures that all bond dimensions are constant
            throughout the MPS, i.e. mps[0].dim = (1 x d[0] x mbd)
            instead of mps[0].dim = (1 x d[0] x d[0]), and so forth.
        subdir : str
            A name for the subdirectory, under CALCDIR specified in 
            cyclomps.tools.params, where this mpsList will be stored

    Returns:
        mps : list
            A python list containing a single MPS, which
            contains the filenames where the individual
            tensors are located
    """

    mpiprint(3,'Creating Random MPS for state {}'.format(state))

    # Number of MPS sites
    N = len(d)
    
    # Allocate list to hold MPS
    mps = []

    # Loop through sites, creating MPS at each site
    for site in range(N):
        
        mpiprint(4,'\tAt Site {}'.format(site))

        # Find required tensor dimensions
        dims = calc_site_dims(site,d,mbd,
                              periodic=periodic,
                              fixed_bd=fixed_bd)
        
        # Specify file location & name
        fname = loc + '/state'+str(state)+'_site'+str(site)
        tenDic = {'dims': dims, 'fname': fname, 'dtype': dtype}
        mps.append(tenDic)
        
        # Create CTF/np array
        tenTmp = rand(dims,dtype=dtype)
        
        # Save CTF/np array at given location
        save_ten(tenTmp,tenDic['fname'])

    return mps 

def mps_load_ten(mpsList,state,site):
    """
    Load a tensor from an MPS
    
    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        state : int
            The state of interest
        site : int
            The site of interest
    """
    # Get Useful info from mpsList
    fname = mpsList[state][site]['fname']
    mpiprint(8,'Loading mps tensor at {}'.format(fname))
    dims = mpsList[state][site]['dims']
    dtype = mpsList[state][site]['dtype']

    # Load Tensor
    ten = load_ten(dims,fname,dtype=dtype)
    return ten

def mps_save_ten(ten,mpsList,state,site):
    """
    Save a tensor from an MPS
    
    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        state : int
            The state of interest
        site : int
            The site of interest
    """
    # Get Useful info from mpsList
    fname = mpsList[state][site]['fname']

    # Load Tensor
    save_ten(ten,fname)
    # PH - Try to delete ten to free up memory

def calc_site_dims(site,d,mbd,
                   periodic=False,fixed_bd=False):
    """
    Calculate the dimensions for an mps tensor

    Args:
        site : int
            The current site in the mps
        d : 1D Array
            A list of the local state space dimension
            for the system
        site_d : int
            Local bond dimension at current site
        mbd : int
            Maximum retained bond dimension

    Kwargs:
        periodic : bool
            If this is true, then create a list of periodic
            mpss, meaning that left and right tensors have 
            dimension [d x mbd x mbd]. If this is False,
            these have dimension [d x 1 x mbd] and 
            [d x mbd x 1]. 
        fixed_bd : bool
            If using open boundary conditions (i.e. periodic==False)
            this ensures that all bond dimensions are constant
            throughout the MPS, i.e. mps[0].dim = (1 x d[0] x mbd)
            instead of mps[0].dim = (1 x d[0] x d[0]), and so forth.

    Returns:
        dims : 1D Array
            A list of len(dims) == 3, with the dimensions of the
            tensor at the current site.
    """
    
    # Find lattice size
    N = len(d)

    # Current limitation: d is symmetric
    for i in range(N):
        assert d[i] == d[N-(i+1)],'Current limitation: d must be symmetric around center'

    # Find local bond dimension
    dloc = d[site]
    
    # Periodic Case (simplest)
    if periodic: 
        dims = [mbd,dloc,mbd]
        return dims

    # First Site (special case)
    if site == 0:
        if fixed_bd:
            dims = [1,dloc,mbd]
        else:
            dims = [1,dloc,dloc]
    
    # Last Site (special Case)
    elif site == N-1:
        if fixed_bd:
            dims = [mbd,dloc,1]
        else:
            dims = [dloc,dloc,1]

    # Central Site (general case)
    else:
        if fixed_bd:
            dims = [mbd,dloc,mbd]
        else:
            if site < int(N/2):
                diml = min(mbd,prod((d[:site])))
                dimr = min(mbd,prod(d[:site+1]))
            elif (site == int(N/2)) and (N%2):
                diml = min(mbd,prod(d[:site]))
                dimr = min(mbd,prod(d[site+1:]))
            else:
                diml = min(mbd,prod(d[site:]))
                dimr = min(mbd,prod(d[site+1:]))
            dims = [diml,dloc,dimr]

    return dims

def calc_entanglement(S):
    """
    Calculate entanglement given a vector of singular values

    Args:
        S : 1D Array
            Singular Values

    Returns:
        EE : double
            The von neumann entanglement entropy
        EEspec : 1D Array
            The von neumann entanglement spectrum
            i.e. EEs[i] = S[i]^2*log_2(S[i]^2)
    """
    # Ensure correct normalization
    norm_fact = sqrt(dot(S,conj(S)))
    S /= norm_fact

    # Calc Entanglement Spectrum
    EEspec = -S*conj(S)*log2(S*conj(S))

    # Sum to get Entanglement Entropy
    EE = summ(EEspec)

    # Print Results
    mpiprint(8,'Entanglement Entropy = {}'.format(EE))
    mpiprint(9,'Entanglement Spectrum = {}'.format(EEspec))

    # Return Results
    return EE,EEspec

def calc_tens_ent(ten,direction):
    """
    Calculate the entanglement entropy from a single mps
    tensor (which we assume to contain the center gauge)

    Args:
        ten : ctf or np array
            A single 3-legged tensor containing the 
            center gauge for the mps
        direction : str
            If 'left', then do svd to the left, if 'right',
            then do svd to the right.

    Returns:
        EE : double
            The von neumann entanglement entropy
        EEspec : 1D Array
            The von neumann entanglement spectrum
            i.e. EEs[i] = S[i]^2*log_2(S[i]^2)
    """
    if direction.lower == 'left':
        _,S,_ = svd(ten,1)
    elif direction.lower == 'right':
        _,S,_ = svd(ten,2)
    return calc_entanglement(S)

def svd_ten(ten,split_ind,mbd=None,return_ent=True,return_wgt=True):
    """
    Compute the Singular Value Decomposition of an input tensor

    Args:
        ten : ctf or np array
            Array for which the svd will be done
        split_ind : int
            The dimension where the split into a matrix will be made

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True
        mbd : int
            The Maximum retained Bond Dimension

    Returns:
        U : ctf or np array
            The resulting U matrix from the svd
        S : ctf or np array
            The resulting singular values from the svd
        V : ctf or np array
            The resulting V matrix from the svd
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    # Reshape tensor into matrix
    ten_shape = ten.shape
    ten = ten.reshape([prod(ten_shape[:split_ind]),prod(ten_shape[split_ind:])])

    # Perform svd
    U,S,V = svd(ten)

    # Compute Entanglement
    EE,EEs = calc_entanglement(S)

    # Truncate results (if necessary)
    D = S.shape[0]

    # Make sure D is not larger than allowed
    if mbd is not None:
        D = min(D,mbd)

    # Compute amount of S discarded
    wgt = S[D:].sum()

    # Truncate U,S,V
    U = U[:,:D]
    S = S[:D]
    V = V[:D,:]

    # Reshape to match correct tensor format
    U = U.reshape(ten_shape[:split_ind]+(-1,))
    V = V.reshape((-1,)+ten_shape[split_ind:])

    # Print some results
    mpiprint(4,'Entanglement Entropy = {}'.format(EE))
    mpiprint(7,'EE Spectrum = ')
    nEEs = EEs.shape[0]
    for i in range(nEEs):
        mpiprint(7,'   {}'.format(EEs[i]))
    mpiprint(5,'Discarded weights = {}'.format(wgt))

    # Return results
    if return_wgt and return_ent:
        return U,S,V,EE,EEs,wgt
    elif return_wgt:
        return U,S,V,wgt
    elif return_ent:
        return U,S,V,EE,EEs
    else:
        return U,S,V

def move_gauge_right_tens(ten1,ten2,return_ent=True,return_wgt=True):
    """
    Move the gauge via svd from ten1 to ten2

    Args:
        ten1 : np or ctf array
            The site currently holding the gauge
        ten2 : np or ctf array
            The neighboring right site

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True

    Returns:
        ten1 : np or ctf array
            The now isometrized tensor
        ten2 : np or ctf array
            The tensor now holding the gauge
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    (n1,n2,n3) = ten1.shape
    (n4,n5,n6) = ten2.shape

    # Perform the svd on the tensor
    U,S,V,EE,EEs,wgt = svd_ten(ten1,2,mbd=None)

    # Pad result to keep bond dim fixed
    Upad = zeros((n1,n2,n3),dtype=type(U[0,0,0]))
    Upad[:,:,:min(n1*n2,n3)] = U
    Spad = zeros((n3),dtype=type(S[0]))
    Spad[:min(n1*n2,n3)] = S
    Vpad = zeros((n3,n4),dtype=type(V[0,0]))
    Vpad[:min(n1*n2,n3),:] = V
    ten1 = Upad

    # Multiply remainder into neighboring site
    gauge = einsum('a,ab->ab',Spad,Vpad)
    ten2 = einsum('ab,bpc->apc',gauge,ten2)

    # Return results
    if return_wgt and return_ent:
        return ten1,ten2,EE,EEs,wgt
    elif return_wgt:
        return ten1,ten2,wgt
    elif return_ent:
        return ten1,ten2,EE,EEs
    else:
        return ten1,ten2

def move_gauge_right(mpsList,state,site,return_ent=True,return_wgt=True):
    """
    Move the gauge one site to the right

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        state : int
            The state which will be one step closer to left canonicalized
        site : int
            The site that currently contains the gauge

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True

    Returns:
        mpsList : 1D Array of Matrix Product States
            The input MPS with the gauge moved one site to the left
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    mpiprint(5,'\t\tMoving Gauge from site {} to {}'.format(site,site+1))

    # Load tensor at given site
    ten1 = mps_load_ten(mpsList,state,site)
    ten2 = mps_load_ten(mpsList,state,site+1)

    # Perform svd
    ten1,ten2,EE,EEs,wgt = move_gauge_right_tens(ten1,ten2)

    # Save the results
    mps_save_ten(ten1,mpsList,state,site)
    mps_save_ten(ten2,mpsList,state,site+1)

    # Return results
    if return_ent and return_wgt:
        return mpsList,EE,EEs,wgt
    elif return_ent:
        return mpsList,EE,EEs
    elif return_wgt:
        return mpsList,wgt
    else:
        return mpsList

def move_gauge_left_tens(ten1,ten2,return_ent=True,return_wgt=True):
    """
    Move the gauge via svd from ten2 to ten1

    Args:
        ten1 : np or ctf array
            The neighboring left site
        ten2 : np or ctf array
            The site currently holding the gauge

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True

    Returns:
        ten1 : np or ctf array
            The tensor now holding the gauge
        ten2 : np or ctf array
            The now isometrized tensor
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    (n1,n2,n3) = ten1.shape
    (n4,n5,n6) = ten2.shape

    # Perform the svd on the tensor
    U,S,V,EE,EEs,wgt = svd_ten(ten2,1,mbd=None)

    # Pad result to keep bond dim fixed
    Vpad = zeros((n4,n5,n6),dtype=type(V[0,0,0]))
    Vpad[:min(n4,n5*n6),:,:] = V
    Spad = zeros((n4),dtype=type(S[0]))
    Spad[:min(n4,n5*n6)] = S
    Upad = zeros((n3,n4),dtype=type(U[0,0]))
    Upad[:,:min(n4,n5*n6)] = U
    ten2 = Vpad

    # Multiply remainder into neighboring site
    gauge = einsum('ab,b->ab',Upad,Spad)
    ten1 = einsum('apb,bc->apc',ten1,gauge)
    
    # Return results
    if return_wgt and return_ent:
        return ten1,ten2,EE,EEs,wgt
    elif return_wgt:
        return ten1,ten2,wgt
    elif return_ent:
        return ten1,ten2,EE,EEs
    else:
        return ten1,ten2

def move_gauge_left(mpsList,state,site,return_ent=True,return_wgt=True):
    """
    Move the gauge one site to the left
    
    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        state : int
            The state which will be right canonicalized
        site : int
            The site that currently contains the gauge

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True

    Returns:
        mpsList : 1D Array of Matrix Product States
            The input MPS with the gauge moved one site to the left
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True

    """
    mpiprint(5,'\t\tMoving Gauge from site {} to {}'.format(site,site-1))

    # Load tensor at given site and neighbor
    ten1 = mps_load_ten(mpsList,state,site-1)
    ten2 = mps_load_ten(mpsList,state,site)

    # Transfer the gauge
    ten1,ten2,EE,EEs,wgt = move_gauge_left_tens(ten1,ten2)
    
    # Save the results
    mps_save_ten(ten1,mpsList,state,site-1)
    mps_save_ten(ten2,mpsList,state,site)

    # Return results
    if return_ent and return_wgt:
        return mpsList,EE,EEs,wgt
    elif return_ent:
        return mpsList,EE,EEs
    elif return_wgt:
        return mpsList,wgt
    else:
        return mpsList

def move_all_gauge_left(mpsList,site,return_ent=True,return_wgt=True):
    """
    Move the gauge one site to the left for all states in the mps list
    
    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        site : int
            The site that currently contains the gauge

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True

    Returns:
        mpsList : 1D Array of Matrix Product States
            The input MPS with the gauge moved one site to the left
        EE : 1D list of floats
            The von neumann entanglement entropy for each state
            Only returned if return_ent == True
        EEs : 1D list of 1D arrays of floats
            The von neumann entanglement spectrum for each state
            Only returned if return_ent == True
        wgt : 1D list of floats
            The sum of the discarded weigths for each state
            Only returned if return_wgt == True

    """
    mpiprint(5,'\t\tMoving Gauge for all states from site {} to site {}'.format(site,site-1))
    # Get mps info
    nStates = len(mpsList)

    # Move gauge for each state
    EE = [None]
    EEs = [None]
    wgt = [None]
    for state in range(nStates):
        mpsList,EE_,EEs_,wgt_ = move_gauge_left(mpsList,state,site)
        EE[state] = EE_
        EEs[state] = EEs_
        wgt[state] = wgt_

    # Return Results 
    if return_ent and return_wgt:
        return mpsList,EE,EEs,wgt
    elif return_ent:
        return mpsList,EE,EEs
    elif return_wgt:
        return mpsList,wgt
    else:
        return mpsList

def move_all_gauge_right(mpsList,site,return_ent=True,return_wgt=True):
    """
    Move the gauge one site to the right for all states in the mps list
    
    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        site : int
            The site that currently contains the gauge

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True

    Returns:
        mpsList : 1D Array of Matrix Product States
            The input MPS with the gauge moved one site to the left
        EE : 1D list of floats
            The von neumann entanglement entropy for each state
            Only returned if return_ent == True
        EEs : 1D list of 1D arrays of floats
            The von neumann entanglement spectrum for each state
            Only returned if return_ent == True
        wgt : 1D list of floats
            The sum of the discarded weigths for each state
            Only returned if return_wgt == True

    """

    mpiprint(5,'\t\tMoving Gauge for all states from site {} to site {}'.format(site,site+1))
    # Get mps info
    nStates = len(mpsList)

    # Move gauge for each state
    EE = [None]
    EEs = [None]
    wgt = [None]
    for state in range(nStates):
        mpsList,EE_,EEs_,wgt_ = move_gauge_right(mpsList,state,site)
        EE[state] = EE_
        EEs[state] = EEs_
        wgt[state] = wgt_

    # Return Results 
    if return_ent and return_wgt:
        return mpsList,EE,EEs,wgt
    elif return_ent:
        return mpsList,EE,EEs
    elif return_wgt:
        return mpsList,wgt
    else:
        return mpsList

def make_mps_right(mpsList,state):
    """
    Make the MPS right canonical

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        state : int
            The state which will be right canonicalized

    Kwargs:
        norm : string
            If norm is not None, then the MPS will be normalized
            at each site to 1 with the L1 norm if norm == 'L1' 
            or the L2 norm if norm == 'L2'
            Default : norm = None

    Returns:
        mpsList : 1D Array of Matrix Product States
            A right canonicalized version of the input MPS
    """
    mpiprint(5,'\tRight Canonicalizing state {}'.format(state))
    nState = len(mpsList)
    N = len(mpsList[0])
    # Loop backwards 
    for site in range(int(N)-1,0,-1):
        mpsList = move_gauge_left(mpsList,state,site,return_ent=False,return_wgt=False)
    return mpsList

def make_mps_list_right(mpsList):
    """
    Make all the MPS in a list right canonical

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 

    Kwargs:
        norm : string
            If norm is not None, then the MPS will be normalized
            at each site to 1 with the L1 norm if norm == 'L1' 
            or the L2 norm if norm == 'L2'
            Default : norm = None

    Returns:
        mpsList : 1D Array of Matrix Product States
            A right canonicalized version of the input MPS
    """
    mpiprint(5,'Making all mps right canonical')
    nStates = len(mpsList)
    for state in range(nStates):
        mpsList = make_mps_right(mpsList,state)
    return mpsList

def move_gauge(mpsList,gauge_init,gauge_fin):
    """
    Move the gauge from the site gauge_init
    to the site gauge_fin

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        gauge_init : int
            The current site where the gauge is located
        gauge_init : int
            The site where the gauge is needed

    Returns:
        mpsList : 1D Array of Matrix Product States
            The correctly gauged version of the MPS
    """
    if gauge_init > gauge_fin:
        # Must move left
        for site in range(gauge_init-1,gauge_fin,-1):
            mpsList = move_all_gauge_left(mpsList,site)
    elif gauge_init < gauge_fin:
        # Must move left
        for site in range(gauge_init,gauge_fin):
            mpsList = move_all_gauge_right(mpsList,site)
    return mpsList

def mps_conj(mpsList,copy=False,copy_subdir='mps_conj'):
    """
    Conjugate an mpsList
    
    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 

    Kwargs:
        copy : bool
            Indicate whether to create a copy of the 
            mps.
            Default : False
        copy_subdir : string
            If a copy is created, indicates the subdirectory
            where the copy will be saved.
            Default : mps_conj

    Returns:
        mpsList : 1D Array of Matrix Product States
            The conjugated version of the input mps list
    """
    mpiprint(6,'Conjugating MPS List')

    # Copy if desired
    if copy: mpsList = mps_copy(mpsList,subdir=copy_subdir)
    
    # Get MPS List Size
    nState = len(mpsList)
    nSite  = len(mpsList[0])

    # Loop through and conjugate each site
    for state in range(nState):
        for site in range(nSite):
            mpiprint(7,'Conjugating tensor at state {} site {}'.format(state,site))
            ten = mps_load_ten(mpsList,state,site)
            ten = conj(ten)
            mps_save_ten(ten,mpsList,state,site)

def mps_copy(mpsList,subdir='mps'):
    """
    Create a copy of an mpsList

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 

    Returns:
        mpsList : 1D Array of Matrix Product States
            A copy of the input mps list
    """
    mpiprint(8,'Copying MPS List')

    # Get MPS List Size
    nState = len(mpsList)
    nSite = len(mpsList[0])
    
    # Create required subdirectory
    if not (subdir[0] == '/'): subdir = '/'+subdir
    subdir_ind = 0
    created = False
    while not created:
        if not os.path.exists(CALCDIR+subdir+str(subdir_ind)):
            os.mkdir(CALCDIR+subdir+str(subdir_ind))
            subdir = subdir + str(subdir_ind)
            created = True
        subdir_ind += 1

    # Loop through states and sites, copying files at each
    mpsListCopy = []
    for state in range(nState):
        mps = []
        for site in range(nSite):
            mpiprint(9,'\tCopying State {} Site {}'.format(state,site))
        
            # Get new file name
            old_fname = mpsList[state][site]['fname']
            new_fname = CALCDIR + subdir + '/state'+str(state)+'_site'+str(site)
        
            # Copy Old file
            try:    
                copyfile(old_fname,new_fname)
            except:
                copyfile(old_fname+'.npy',new_fname+'.npy')

            # Update new MPS
            dims = mpsList[state][site]['dims']
            dtype= mpsList[state][site]['dtype']
            tenDic = {'dims': dims, 'fname': old_fname, 'dtype': dtype}
            
            # Add to mps
            mps.append(tenDic)

        # Add to mps List
        mpsListCopy.append(mps)

    return mpsListCopy

def contract_config(mpsList,config,norm=None,state=0,gSite=0):
    """
    Contract the mps to give the coefficient for a given configurations

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        config : 1D Array
            Corresponds to the configuration of interest,
            i.e. for a spin-1/2 model with N=8, this might be 
            [0,1,1,0,0,1,1,0]

    Kwargs:
        norm : str
            'L1' -> Use L1 Norm
            'L2' -> Use L2 Norm
            None -> Use current normalization
        state : int
            The state of interest, default is 0
        gSite : int
            The site where the gauge is located, allows normalization
            without the full MPS contraction

    Returns:
        val : float
            The probability of being in the given configuration
    """
    mpiprint(4,'Contraction MPS for configuration {}'.format(config))

    # Get MPS List Size
    nState = len(mpsList)
    nSite  = len(mpsList[0])

    # Check Normalization
    gTen = mps_load_ten(mpsList,state,gSite)
    if norm == 'L1':
        mps[state][gSite] /= einsum('apb->',gTen)
    elif norm == 'L2':
        mps[state][gSite] /= einsum('apb,apb->',gTen,conj(gTen))

    # Contract MPS for given config
    ten1 = mps_load_ten(mpsList,state,0)
    ten1 = ten1[:,config[0],:]
    for site in range(1,nSite):
        ten2 = mps_load_ten(mpsList,state,site)
        ten2 = ten2[:,config[site],:]
        ten1 = einsum('ab,bc->ac',ten1,ten2)
    
    # Contract once again (just in case its periodic)
    val = einsum('ab->',ten1) # PH - Should this be aa-> or ab-> ???

    return val


def increase_bond_dim(mpsList,mbd,
                      copy=True,fixed_bd=False,
                      copy_subdir='mps',periodic=False):
    """
    Create a copy of an mpsList

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 

    Kwargs:
        copy : bool
            Indicate whether to create a copy of the 
            mps.
            Default : False
        copy_subdir : string
            If a copy is created, indicates the subdirectory
            where the copy will be saved.
            Default : mps_conj
        fixed_bd : bool
            If using open boundary conditions (i.e. periodic==False)
            this ensures that all bond dimensions are constant
            throughout the MPS, i.e. mps[0].dim = (1 x d[0] x mbd)
            instead of mps[0].dim = (1 x d[0] x d[0]), and so forth.

    Returns:
        mpsList : 1D Array of Matrix Product States
            A copy of the input mps list
    """
    mpiprint(8,'Copying MPS List')
    
    # Get info from mpsList
    nState = len(mpsList)
    nSite = len(mpsList[0])
    d = mps_local_dim(mpsList)

    # Copy state if desired
    if copy: mpsList = mps_copy(mpsList,subdir=copy_subdir)

    # Loop through all states
    for state in range(nState):

        # Loop through all sites
        for site in range(nSite):

            # Load the old tensor
            old_ten = mps_load_ten(mpsList,state,site)
            dtype = mpsList[state][site]['dtype']
            dim_old = mpsList[state][site]['dims']

            # Determine dims of new tensor
            dims = calc_site_dims(site,d,mbd,
                                  periodic=periodic,
                                  fixed_bd=fixed_bd)
            mpsList[state][site]['dims'] = dims
            
            # Create new CTF/np array
            new_ten = zeros(dims,dtype=dtype)

            # Fill in old tensor
            new_ten[:dim_old[0],:dim_old[1],:dim_old[2]] = old_ten

            # Save CTF/np array at given location
            save_ten(new_ten,mpsList[state][site]['fname'])
            
    # Return the resulting tensor
    return mpsList

def mps_local_dim(mpsList):
    """ 
    Determine the local bond dimensions from an mpsList

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 

    Returns:
        d : 1D array
            A list corresponding to the local state space dimension
            of each site, as determined by the mpo
    """
    mpiprint(7,'Determining local bond dimensions of mps')

    # Get info from mpo list
    nStates = len(mpsList)
    nSites = len(mpsList[0])

    # Create list of local bond dims
    d = [0]*nSites

    # Loop through all sites in the first state
    state = 0
    for site in range(nSites):
        _,dtmp,_ = mpsList[state][site]['dims']
        d[site] = dtmp

    # Return Results
    return d
