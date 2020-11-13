"""
Tools for Matrix Product State Environments

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

.. To Do:
    -Add environment tools for iDMRG procedure

"""

from cyclomps.tools.utils import *
from cyclomps.tools.mps_tools import mps_load_ten

def alloc_env(mpsL,mpoL,mpslL=None,dtype=None,subdir='env'):
    """ 
    Allocate tensors for the dmrg environment tensors

    Args:
        mpsL : 1D Array
            The matrix product state list, generate with
            cyclomps.tools.mps_tools.create_mps_list(), 
            for which the environment is being created.
        mpoL : 1D Array
            A list of mpos, whose structure is described in 
            cyclomps.mpos, with some examples there

    Kwargs:
        mpslL : 1D Array
            The matrix product state list for the ket state,
            for which the environment is being created

    Returns:
        envL : 1D Array
            An environment
    """
    mpiprint(4,'Allocating environment')

    # Create required subdirectory
    if not (subdir[0] == '/'): subdir = '/'+subdir
    subdir_ind = 0 
    created = False
    while not created:
        if not os.path.exists(CALCDIR+subdir+str(subdir_ind)):
            mkdir(CALCDIR+subdir+str(subdir_ind))
            subdir = subdir + str(subdir_ind)
            created = True
        subdir_ind += 1
    loc = CALCDIR+subdir

    # Get details about mpo and mps
    if mpslL is None: mpslL = mpsL
    nState = len(mpsL)
    nSite  = len(mpsL[0])
    nOps   = len(mpoL)

    # Initialize empty list to hold envL
    envL = []
    
    # Loop over all operators in mpo List
    for op in range(nOps):
        # Create an environment (in a list) for given operator
        env = []

        # Initial entry (for edge) PH - Does not work for periodic
        dims = (1,1,1)
        ten = zeros(dims,dtype=dtype)
        ten[0,0,0] = 1.
        fname = loc + '/op'+str(op)+'_site'+str(0)
        envDic = {'dims': dims, 'fname': fname, 'dtype': dtype}
        # Save Tensor
        save_ten(ten,envDic['fname'])
        # Add to env list
        env.append(envDic)
        
        # Central Entries 
        for site in range(nSite-1):
            # Find required dimensions
            mps_D = mpsL[0][site]['dims'][2]
            mpsl_D= mpslL[0][site]['dims'][2]
            for op in range(nOps):
                mpo_D=1
                if mpoL[op][site] is not None:
                    _,_,_,mpo_D = mpoL[op][site].shape
            # Create tensor
            dims = (mpsl_D,mpo_D,mps_D)
            ten = zeros(dims,dtype=dtype)
            fname = loc + '/op'+str(op)+'_site'+str(site+1)
            envDic = {'dims': dims, 'fname': fname, 'dtype': dtype}
            # Save Tensor
            save_ten(ten,envDic['fname'])
            # Add to env list
            env.append(envDic)

        # Final entry (for right edge) PH - Does not work for periodic
        dims = (1,1,1)
        ten = zeros(dims,dtype=dtype)
        ten[0,0,0] = 1.
        fname = loc + '/op'+str(op)+'_site'+str(nSite)
        envDic = {'dims': dims, 'fname': fname, 'dtype': dtype}
        # Save Tensor
        save_ten(ten,envDic['fname'])
        # Add to env list
        env.append(envDic)

        # Add new env to the overall env list
        envL.append(env)

    # Return list of environments
    return envL

def load_env(envList,op):
    """
    Load a full environment and return it as a list
    """
    nOp = len(envList)
    N = len(envList[op])
    env = [None]*N
    for site in range(N):
        env[site] = env_load_ten(envList,op,site)
    return env

def load_env_list(envList):
    """
    Load a full environment list and return it as a list of environments
    """
    nOp = len(envList)
    env = [None]*nOp
    for op in range(nOp):
        env[op] = load_env(envList,op)
    return env

def env_load_ten(envList,op,site):
    """
    Load a tensor from an env list

    Args:
        envList : 1D Array of an environment
            A list containing multiple environments,
            each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local env tensor. 
        state : int
            The state of interest
        site : int
            The site of interest
    """
    # Get Useful info from mpsList
    fname = envList[op][site]['fname']
    dims = envList[op][site]['dims']
    dtype = envList[op][site]['dtype']
    # Load Tensor
    ten = load_ten(dims,fname,dtype=dtype)
    return ten

def env_save_ten(ten,envList,op,site):
    """ 
    Save a tensor from an MPS

    Args:
        ten : np or ctf array
            The tensor to be saved
        envList : 1D Array of an environment
            A list containing multiple environments,
            each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local env tensor. 
        op : int
            The mpo of interest
        site : int
            The site of interest
    """
    # Get Useful info from mpsList
    fname = envList[op][site]['fname']
    # Load Tensor
    save_ten(ten,fname)
    # PH - Try to delete ten to free up memory

def update_env_left(mpsList,mpoList,envList,site,mpslList=None,state=0):
    """
    Update the environment tensors as we move to the left

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        mpoList : 1D Array of MPOs
            A list containing multiple MPOs
        envList : 1D Array of an environment
            A list containing multiple environments,
            each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local env tensor. 
        site : int
            The current site

    Kwargs:
        mpslList : 1D Array of Matrix Product State
            An optional mpsList representing the left state of the 
            system if you would like left and right vectors
            used in computing the environment
        state : int
            Default : 0
            The state that will be used in updating the environment.
            Usually this is not important when using state averaging,
            because all left normalized states have the same tensor
            at the given site

    Returns:
        envList : 1D Array of an environment
            This returns the same environmnet you started with, though
            the entries have been updated
    """
    mpiprint(4,'\tUpdating Environment moving from site {} to {}'.format(site,site-1))

    # Get useful info
    nOp = len(mpoList)
    nState = len(mpsList)
    nSite = len(mpsList[0])

    # Get a left MPS, 
    if mpslList is None: mpslList = mpsList
    
    # Loop through all operators
    for op in range(nOp):
        # Load environment and mps
        envLoc = env_load_ten(envList,op,site+1)
        mpsLoc = mps_load_ten(mpsList,state,site)
        mpslLoc= mps_load_ten(mpslList,state,site) # PH - Using extra memory here

        # Do Contraction to update env
        if mpoList[op][site] is None:
            envNew = einsum('apb,Bcb->apcB',mpsLoc,envLoc)
            envNew = einsum('apcB,ApB->Aca',envNew,conj(mpslLoc))
        else:
            envNew = einsum('apb,Bcb->apcB',mpsLoc,envLoc)
            envNew = einsum('apcB,dqpc->adqB',envNew,mpoList[op][site]) # PH - Might be dpqc?
            envNew = einsum('adqB,AqB->Ada',envNew,conj(mpslLoc))

        # Save resulting new environment
        env_save_ten(envNew,envList,op,site)
    
    # Return environment list
    return envList

def update_env_left_local(mpsLoc,mpoLoc,envLoc,mpslLoc=None,ts=0):
    """
    """
    mpiprint(4,'Local environment update')

    # Get useful info
    nOp = len(mpoLoc)
    nState = len(mpsLoc)

    # Get a left MPS, 
    if mpslLoc is None: mpslLoc = mpsLoc
    
    # Loop through all operators
    for op in range(nOp):

        # Do Contraction to update env
        if mpoLoc[op] is None:
            envNew = einsum('apb,Bcb->apcB',mpsLoc[ts],envLoc[op])
            envNew = einsum('apcB,ApB->Aca',envNew,conj(mpslLoc[ts]))
        else:
            envNew = einsum('apb,Bcb->apcB',mpsLoc[ts],envLoc[op])
            envNew = einsum('apcB,dqpc->adqB',envNew,mpoLoc[op]) # PH - Might be dpqc?
            envNew = einsum('adqB,AqB->Ada',envNew,conj(mpslLoc[ts]))

        # Put result into local env list
        envLoc[op] = envNew

    # Return environment list
    return envLoc

def update_env_right_local(mpsLoc,mpoLoc,envLoc,mpslLoc=None,ts=0):
    """
    """
    mpiprint(4,'Local environment update')

    # Get useful info
    nOp = len(mpoLoc)
    nState = len(mpsLoc)

    # Get a left MPS, 
    if mpslLoc is None: mpslLoc = mpsLoc
    
    # Loop through all operators
    for op in range(nOp):
        mpiprint(8,'Operator {}/{}'.format(op,nOp))
        # Do Contraction to update env
        if mpoLoc[op] is None:
            envNew = einsum('Bcb,BqA->cbqA',envLoc[op],conj(mpslLoc[ts]))
            envNew = einsum('bpa,cbpA->Aca',mpsLoc[ts],envNew)
        else:
            envNew = einsum('Bcb,BqA->cbqA',envLoc[op],conj(mpslLoc[ts]))
            envNew = einsum('cqpd,cbqA->pdbA',mpoLoc[op],envNew)
            envNew = einsum('bpa,pdbA->Ada',mpsLoc[ts],envNew)

        # Put result into local env list
        envLoc[op] = envNew
    
    # Return environment list
    return envLoc

def update_env_right(mpsList,mpoList,envList,site,mpslList=None,state=0):
    """
    Update the environment tensors as we move to the right

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        mpoList : 1D Array of MPOs
            A list containing multiple MPOs
        envList : 1D Array of an environment
            A list containing multiple environments,
            each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local env tensor. 
        site : int
            The current site

    Kwargs:
        mpslList : 1D Array of Matrix Product State
            An optional mpsList representing the left state of the 
            system if you would like left and right vectors
            used in computing the environment
        state : int
            Default : 0
            The state that will be used in updating the environment.
            Usually this is not important when using state averaging,
            because all left normalized states have the same tensor
            at the given site

    Returns:
        envList : 1D Array of an environment
            This returns the same environmnet you started with, though
            the entries have been updated
    """
    mpiprint(4,'\tUpdating Environment moving from site {} to {}'.format(site,site+1))

    # Get useful info
    nOp = len(mpoList)
    nState = len(mpsList)
    nSite = len(mpsList[0])

    # Get a left MPS, 
    if mpslList is None: mpslList = mpsList
    
    # Loop through all operators
    for op in range(nOp):
        # Load environment and mps
        envLoc = env_load_ten(envList,op,site)
        mpsLoc = mps_load_ten(mpsList,state,site)
        mpslLoc= mps_load_ten(mpslList,state,site) # PH - Using extra memory here

        # Do Contraction to update env
        if mpoList[op][site] is None:
            envNew = einsum('Ala,APB->alPB',envLoc,conj(mpslLoc))
            envNew = einsum('aPb,alPB->Blb',mpsLoc,envNew)
        else:
            envNew = einsum('Ala,APB->alPB',envLoc,conj(mpslLoc))
            envNew = einsum('lPpr,alPB->aprB',mpoList[op][site],envNew)
            envNew = einsum('apb,aprB->Brb',mpsLoc,envNew)

        # Save resulting new environment
        env_save_ten(envNew,envList,op,site+1)
    
    # Return environment list
    return envList

def calc_env(mpsList,mpoList,mpslList=None,dtype=None,subdir='env',gSite=0,state=0):
    """
    Calculate all the environment tensors

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        mpoList : 1D Array of MPOs
            A list containing multiple MPOs

    Kwargs:
        mpslList : 1D Array of Matrix Product State
            An optional mpsList representing the left state of the 
            system if you would like left and right vectors
            used in computing the environment
        dtype : dtype
            Specify the data type, if None, then this will use the 
            same datatype as the mpsList.
        subdir : string
            The subdirectory within the current calculation directory
            where the env will be saved.
        gSite : int
            The site where the gauge is currently located. The default
            is gSite=0, meaning the mps is right canonical.
        state : int
            Default : 0
            The state that will be used in updating the environment.
            Usually this is not important when using state averaging,
            because all left normalized states have the same tensor
            at the given site

    Returns:
        envList : 1D Array of an environment
            This returns the same environmnet you started with, though
            the entries have been updated
    """
    mpiprint(4,'Calculating full environment')
    nSite = len(mpsList[0])
    # Make env same dtype as mps
    if dtype is None: dtype = mpsList[0][0]['dtype']
    # Allocate Environment
    envList = alloc_env(mpsList,mpoList,mpslL=mpslList,dtype=dtype,subdir=subdir)
    # Calculate Environment from the right
    for site in range(nSite-1,gSite,-1):
        envList = update_env_left(mpsList,mpoList,envList,site,mpslList=mpslList,state=state)
    for site in range(gSite):
        envList = update_env_right(mpsList,mpoList,envList,site,mpslList=mpslList,state=state)
    return envList
