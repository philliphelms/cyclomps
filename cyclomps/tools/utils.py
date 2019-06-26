"""
Linear Algebra Tools

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""

from cyclomps.tools.params import *
from psutil import virtual_memory as vmem

if USE_CTF:
    from cyclomps.tools.utils_ctf import *
else:
    from cyclomps.tools.utils_np import *

def bytes2human(n):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n

def mpiprint(priority,msg):
    """ 
    Make print only work when RANK == 0, 
    avoiding repeated printing of statements.
    """
    if (RANK == 0) and (priority <= VERBOSE):
        print('  '*priority+msg)

def timeprint(priority,msg):
    """ 
    Make print only work when RANK == 0, 
    avoiding repeated printing of statements.
    """
    if (RANK == 0) and (priority <= VERBOSE_TIME):
        print('  '*priority+msg)

def memprint(priority,msg):
    """ 
    Make print only work when RANK == 0, 
    avoiding repeated printing of statements.
    """
    if (RANK == 0) and (priority <= VERBOSE_MEM):
        tot_mem = bytes2human(vmem()[0])
        av_mem = bytes2human(vmem()[3])
        print('  '*priority+msg+': '+av_mem+' / '+tot_mem)

from cyclomps.tools.mps_tools import mps_load_ten
from cyclomps.tools.mps_tools import mps_save_ten
from cyclomps.tools.env_tools import env_load_ten
from cyclomps.tools.env_tools import env_save_ten

def retrieve_tensors(site,mpsList=None,mpoList=None,envList=None,twoSite=False):
    """
    Retrieve tensors from the mps, mpo and env lists to use in eigenproblem
    """
    mpiprint(5,'Retrieving tensors for eigenproblem')

    # Get info on mps and mpo
    if mpsList is not None: nStates = len(mpsList)
    if mpoList is not None: nOps = len(mpoList)
    if envList is not None: nOps = len(envList)

    # Set up return list
    output = ()

    if mpsList is not None:
        mpiprint(6,'Retrieving mps')
        # Retrieve first mps tensor
        mps_ten1 = []
        for state in range(nStates):
            mps_ten1.append(mps_load_ten(mpsList,state,site))
        output += (mps_ten1,)
        # Retrieve second if needed
        if twoSite:
            mps_ten2 = []
            for state in range(nStates):
                mps_ten2.append(mps_load_ten(mpsList,state,site+1))
            output += (mps_ten2,)

    if mpoList is not None:
        mpiprint(6,'Retrieving mpo')
        # Retrieve first mpo tensor
        mpo_ten1 = []
        for op in range(nOps):
            mpo_ten1.append(mpoList[op][site])
        output += (mpo_ten1,)
        # Retrieve second if needed
        if twoSite:
            mpo_ten2 = []
            for op in range(nOps):
                mpo_ten2.append(mpoList[op][site+1])
            output += (mpo_ten2,)

    if envList is not None:
        mpiprint(6,'Retrieving env')
        # Retrieve First Environment Tensor
        env_left = []
        for op in range(nOps):
            env_left.append(env_load_ten(envList,op,site))
        output += (env_left,)
        # Retrieve Right side environment
        env_right = []
        if twoSite:
            for op in range(nOps):
                env_right.append(env_load_ten(envList,op,site+2))
        else:
            for op in range(nOps):
                env_right.append(env_load_ten(envList,op,site+1))
        output += (env_right,)

    return output

def save_tensors(site,
                 mpsList=None,mps=None,
                 envList=None,env=None):
    """
    Save tensors into the mps and env lists
    """
    mpiprint(5,'Saving tensors from eigenproblem')

    # Get info on mps and mpo
    if mpsList is not None: nStates = len(mpsList)
    if envList is not None: nOps = len(envList)

    # Save mps
    if mpsList is not None:
        mpiprint(6,'Saving mps')
        # Retrieve first mps tensor
        for state in range(nStates):
            mps_save_ten(mps[state],mpsList,state,site)

    # Save env
    if envList is not None:
        mpiprint(6,'Saving env')
        # Retrieve First Environment Tensor
        env_left = []
        for op in range(nOps):
            env_save_ten(env[op],envList,op,site)
