"""
One-site implementation of DMRG algorithm (with state averaging)

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""
from cyclomps.tools.mpo_tools import *
from cyclomps.tools.mps_tools import *
from cyclomps.tools.env_tools import *
from cyclomps.tools.diag_tools import *
from cyclomps.tools.utils import *
from cyclomps.tools.params import *
import scipy.linalg as sla
from numpy import abs as npabs
from numpy import sum as npsum
from numpy import ones as npones
from numpy import complex_
from numpy import zeros as npzeros
from numpy import array as nparray
from numpy import argsort as npargsort
import time
import copy

def calc_rdm(ten,swp_dir):
    """
    Compute the reduced density matrix

    Args: 
        ten : np or ctf tensor
            The site where the gauge is contained in the mps
        swp_dir : string
            The direction of the sweep, either 'left' or 'right'

    Returns: 
        rdm : np or ctf tensor
            The reduced density matrix
    """
    mpiprint(6,'Calculating Reduced density matrix')

    if swp_dir == 'right':
        (n1,n2,n3) = ten.shape
        ten = reshape(ten,(n1*n2,n3))
        return einsum('ij,kj->ik',ten,conj(ten))
    else:
        (n1,n2,n3) = ten.shape
        ten = reshape(ten,(n1,n2*n3))
        return einsum('ij,ik->jk',ten,conj(ten))

def renormalize_right(mps0,mps1,state_avg=True):
    """
    Do the renormalization step to the right, using state averaging

    Args: 
        mps0 : 1D Array
            The mps tensor (list of tensors, one tensor per state)
            for the site currently holding the gauge
        mps1 : 1D Array
            The mps tensor (list of tensors, one tensor per state)
            for the neighboring right site.
    
    Kwargs:
        state_avg : bool
            Specify whether to use the state averaging procedure to target
            multiple states when doing the renormalization step.
            Default : True

    Returns:
        mps0 : 1D Array
            The new mps tensor at the site previously holding 
            the gauge, now an isometry
        mps1 : 1D Array
            The new mps tensor at the neighboring right site,
            which now holds the gauge. 
        EE : 1D List of floats
            The von neumann entanglement entropy for each state
        EEs : 1D List of 1D Array of floats
            The von neumann entanglement spectrum for each state
        wgt : 1D list of floats
            The discarded weight for each state

    """
    if state_avg:
        mpiprint(5,'Renormalize (state avg) right step')
        memprint(5,'  Start Renormalize Memory')
        
        # Determine info from mps0
        nStates = len(mps0)
        (n1,n2,n3) = mps0[0].shape

        # Calculate Entanglement Entropy 
        # PH - Speed up (so we don't calc svd)
        EE = [None]*nStates
        EEs = [None]*nStates
        wgt = [None]*nStates
        for state in range(nStates):
            _,_,EE_,EEs_,wgt_ = move_gauge_right_tens(mps0[state],mps1[state])
            EE[state] = EE_
            EEs[state] = EEs_
            wgt[state] = wgt

        # Compute the reduced density matrix
        w = 1./float(nStates)
        for state in range(nStates):
            if state == 0:
                rdm = w*calc_rdm(mps0[state],'right')
            else:
                rdm+= w*calc_rdm(mps0[state],'right')

        # Take eigenvalues of the rdm
        if USE_CTF: rdm = to_nparray(rdm)
        vals,vecs = sla.eig(rdm)

        # Sort results
        inds = npargsort(vals)[::-1]

        # Retain mbd eigenstates
        inds = inds[:n3]
        vals = vals[inds]
        vecs = vecs[:,inds]

        # Make sure vecs are orthonormal
        vecs = sla.orth(vecs)

        # Put results into mps0 and mps1
        if USE_CTF: vecs = from_nparray(vecs)
        for state in range(nStates):
            # Put resulting eigenvectors into mps0
            mps0_new = zeros((n1,n2,n3),dtype=type(vecs[0,0]))
            mps0_new[:,:,:min(n3,n1*n2)] = reshape(vecs,(n1,n2,min(n3,n1*n2)))

            # Multiply into next site to transfer gauge
            mps1[state] = einsum('apb,apc,cqd->bqd',conj(mps0_new),mps0[state],mps1[state])

            # Write over mps0
            mps0[state] = mps0_new

    # Not state averaged renormalization
    else:
        mpiprint(5,'Renormalize right step')
        memprint(5,'  Start Renormalize Memory')

        # Determine info from mps0
        nStates = len(mps1)
        (n1,n2,n3) = mps1[0].shape

        # Move gauge state-by-state
        EE = [None]*nStates
        EEs = [None]*nStates
        wgt = [None]*nStates
        for state in range(nStates):
            mps0[state],mps1[state],EE_,EEs_,wgt_ = move_gauge_right_tens(mps0[state],mps1[state])
            EE[state] = EE_
            EEs[state] = EEs_
            wgt[state] = wgt

    # Return results
    memprint(5,'  End Renormalize Memory')
    return mps0,mps1,EE,EEs,wgt 

def renormalize_left(mps0,mps1,state_avg=True):
    """
    Do the renormalization step to the left (from mps1 -> mps0)
    using state averaging

    Args: 
        mps0 : 1D Array
            The mps tensor (list of tensors, one tensor per state)
            for the neighboring left site.
        mps1 : 1D Array
            The mps tensor (list of tensors, one tensor per state)
            for the site currently holding the gauge

    Kwargs:
        state_avg : bool
            Specify whether to use the state averaging procedure to target
            multiple states when doing the renormalization step.
            Default : True

    Returns:
        mps0 : 1D Array
            The new mps tensor at the neighboring left site,
            which now holds the gauge. 
        mps1 : 1D Array
            The new mps tensor at the site previously holding 
            the gauge, now an isometry
        EE : 1D List of floats
            The von neumann entanglement entropy for each state
        EEs : 1D List of 1D Array of floats
            The von neumann entanglement spectrum for each state
        wgt : 1D list of floats
            The discarded weight for each state

    """
    if state_avg:
        mpiprint(5,'Renormalize (state avg) left step')
        memprint(5,'  Start Renormalize Memory')
        
        # Determine info from mps0
        nStates = len(mps1)
        (n1,n2,n3) = mps1[0].shape

        # Calculate Entanglement Entropy 
        EE = [None]*nStates
        EEs = [None]*nStates
        wgt = [None]*nStates
        for state in range(nStates):
            mps0[state],mps1[state],EE_,EEs_,wgt_ = move_gauge_left_tens(mps0[state],mps1[state])
            EE[state] = EE_
            EEs[state] = EEs_
            wgt[state] = wgt

        # Compute the reduced density matrix
        w = 1./nStates
        for state in range(nStates):
            if state == 0:
                rdm = w*calc_rdm(mps1[state],'left')
            else:
                rdm+= w*calc_rdm(mps1[state],'left')

        # Take eigenvalues of the rdm
        if USE_CTF: rdm = to_nparray(rdm)
        vals,vecs = sla.eig(rdm)

        # Sort results
        inds = npargsort(vals)[::-1]

        # Retain mbd eigenstates
        inds = inds[:n1]
        vals = vals[inds]
        vecs = vecs[:,inds]

        # Make sure vecs are orthonormal
        vecs = sla.orth(vecs)
        vecs = vecs.T

        # Put results into mps0 and mps1
        if USE_CTF: vecs = from_nparray(vecs)
        for state in range(nStates):
            # Put resulting eigenvectors into mps0
            mps1_new = zeros((n1,n2,n3),dtype=type(vecs[0,0]))
            mps1_new[:min(n1,n2*n3),:,:] = reshape(vecs,(min(n1,n2*n3),n2,n3))

            # Multiply into next site to transfer gauge
            mps0[state] = einsum('apb,bqc,dqc->apd',mps0[state],mps1[state],conj(mps1_new))

            # Write over mps0
            mps1[state] = mps1_new

    # Not state averaged renormalization
    else:
        mpiprint(5,'Renormalize left step')
        memprint(5,'  Start Renormalize Memory')

        # Determine info from mps0
        nStates = len(mps1)
        (n1,n2,n3) = mps1[0].shape

        # Move gauge state-by-state
        EE = [None]*nStates
        EEs = [None]*nStates
        wgt = [None]*nStates
        for state in range(nStates):
            mps0[state],mps1[state],EE_,EEs_,wgt_ = move_gauge_left_tens(mps0[state],mps1[state])
            EE[state] = EE_
            EEs[state] = EEs_
            wgt[state] = wgt

    # Return results
    memprint(5,'  End Renormalize Memory')
    return mps0,mps1,EE,EEs,wgt 

def right_step(site,mps,mpo,env,
               alg='davidson',
               noise=0.,
               orthonormalize=False,
               state_avg=True):
    """
    Perform an optimization and renormalization to the right
    """
    t0 = time.time()
    mpiprint(6,'Start Right Step')
    memprint(6,'  Right Step Start Memory')
    
    # Retrieve relevant tensors
    (mps0,) = retrieve_tensors(site,mpsList=mps)
    (envl,envr) = retrieve_tensors(site,envList=env)
    (mpo,) = retrieve_tensors(site,mpoList=mpo)

    # Solve Eigenproblem
    E,mps0,ovlp = eig1(mps0,mpo,envl,envr,alg=alg)
    mpiprint(3,'Right Step E = {}'.format(E))

    # Perform Renormalization
    (mps1,) = retrieve_tensors(site+1,mpsList=mps)
    mps0,mps1,EE,EEs,wgt = renormalize_right(mps0,mps1,
                                  state_avg=state_avg)

    # Update environment
    envl = update_env_right_local(mps0,mpo,envl)

    # Save results
    save_tensors(site,mpsList=mps,mps=mps0)
    save_tensors(site+1,mpsList=mps,mps=mps1)
    save_tensors(site+1,envList=env,env=envl) # Check index

    # Print Total Sweep time
    timeprint(4,'Right Step Time: {} s'.format(time.time()-t0))
    memprint(6,'  Right Step End Memory')

    # Return results
    return E,EE,EEs,wgt

def left_step(site,mps,mpo,env,
              alg='davidson',
              noise=0.,
              orthonormalize=False,
              state_avg=True):
    """
    Perform an optimization and renormalization to the left

    Args: 
        site : int
            The current site where the optimization 
            should be performed
        mps : 1D Array of Matrix Product States
            The current mps list
            (stored as an mpsList
            as specified in cyclomps.tools.mps_tools)
        mpo : 1D Array
            An array that contains a list of mpos, each of which 
            is a list with an array for each site in the lattice
            representing the mpo at that site.
        env : 1D Array of an environment
            The environemtn for the current mps
            (stored as an envList
            as specified in cyclomps.tools.env_tools)

    Kwargs:
        alg : string
            The algorithm that will be used. Available options are
            'arnoldi', 'exact', and 'davidson', current default is 
            'davidson'.
            Default : 'davidson'
        noise : float or 1D Array of floats
            The magnitude of the noise added to the mbd to prevent
            getting stuck in a local minimum
            !! NOT IMPLEMENTED !!
            Default : 0.
        orthonormalize : bool
            Specify whether to orthonormalize eigenvectors after solution
            of eigenproblem. This will cause problems for all systems
            unless the eigenstates being orthonormalized are degenerate.
            Default : False
        state_avg : bool
            Specify whether to use the state averaging procedure to target
            multiple states when doing the renormalization step.
            Default : True

    Returns:
        E : 1D Array
            The energies for the number of states targeted
        EE : 1D List of floats
            The von neumann entanglement entropy for each state
        EEs : 1D List of 1D Array of floats
            The von neumann entanglement spectrum for each state
        wgt : 1D list of floats
            The discarded weight for each state
    """
    t0 = time.time()
    mpiprint(6,'Start Left Step')
    memprint(6,'  Start Left Step Memory') 
    
    # Retrieve relevant tensors
    (mps0,) = retrieve_tensors(site-1,mpsList=mps)
    (mps1,) = retrieve_tensors(site,mpsList=mps)
    (envl,envr) = retrieve_tensors(site,envList=env)
    (mpo,) = retrieve_tensors(site,mpoList=mpo)

    # Solve Eigenproblem
    E,mps1,ovlp = eig1(mps1,mpo,envl,envr,alg=alg)
    mpiprint(3,'Left Step E = {}'.format(E))

    # Perform Renormalization
    mps0,mps1,EE,EEs,wgt = renormalize_left(mps0,mps1,
                                 state_avg=state_avg)

    # Update environment
    envr = update_env_left_local(mps1,mpo,envr)

    # Save results
    save_tensors(site,mpsList=mps,mps=mps1)
    save_tensors(site-1,mpsList=mps,mps=mps0)
    save_tensors(site,envList=env,env=envr) # PH - Check index

    # Print total sweep time
    timeprint(4,'Left Step Time: {} s'.format(time.time()-t0))
    memprint(6,'  End Left Step Memory')

    # Return results
    return E,EE,EEs,wgt

def right_sweep(mps,mpo,env,
                alg='davidson',
                start_site=None,
                end_site=None,
                noise=0.,
                orthonormalize=False,
                state_avg=True):
    """
    Perform a DMRG sweep from left to right

    Args: 
        mps : 1D Array of Matrix Product States
            The current mps list
            (stored as an mpsList
            as specified in cyclomps.tools.mps_tools)
        mpo : 1D Array
            An array that contains a list of mpos, each of which 
            is a list with an array for each site in the lattice
            representing the mpo at that site.
        env : 1D Array of an environment
            The environemtn for the current mps
            (stored as an envList
            as specified in cyclomps.tools.env_tools)

    Kwargs:
        alg : string
            The algorithm that will be used. Available options are
            'arnoldi', 'exact', and 'davidson', current default is 
            'davidson'.
            Default : 'davidson'
        start_site : int
            The site of the gauge for the input mps, where the DMRG
            sweep will start.
            Default : the left-most site
        end_site : int
            Where the DMRG sweep will end, 
            Default : the right-most site
        noise : float or 1D Array of floats
            The magnitude of the noise added to the mbd to prevent
            getting stuck in a local minimum
            !! NOT IMPLEMENTED !!
            Default : 0.
        orthonormalize : bool
            Specify whether to orthonormalize eigenvectors after solution
            of eigenproblem. This will cause problems for all systems
            unless the eigenstates being orthonormalized are degenerate.
            Default : False
        state_avg : bool
            Specify whether to use the state averaging procedure to target
            multiple states when doing the renormalization step.
            !! ONLY STATE AVG IS IMPLEMENTED !!
            Default : True

    Returns:
        E : 1D Array
            The energies for the number of states targeted
        EE : 1D List of floats
            The von neumann entanglement entropy for each state
        EEs : 1D List of 1D Array of floats
            The von neumann entanglement spectrum for each state
        wgt : 1D list of floats
            The discarded weight for each state
    """
    t0 = time.time()
    mpiprint(2,'\n')
    mpiprint(2,'Beginning Right Sweep')
    mpiprint(2,'-'*50)
    memprint(2,'  Start Right Sweep Memory')
    
    # Get info about mps
    nSite = len(mps[0])
    if start_site is None: start_site = 0
    if end_site is None: end_site = nSite-1

    # Variables to return
    E = None
    EE = None
    EEs = None
    wgt = None

    # Sweep through sites
    for site in range(start_site,end_site):

        # Step left at each site
        E_,EE_,EEs_,wgt_ = right_step(site,mps,mpo,env,
                                      alg=alg,
                                      noise=noise,
                                      orthonormalize=orthonormalize,
                                      state_avg=state_avg)

        # Save results at center site
        if site == int(nSite/2):
            E = E_
            EE = EE_
            EEs = EEs_
            wgt = wgt_

    # Print total sweep time
    timeprint(3,'Right Sweep Time: {} s'.format(time.time()-t0))
    memprint(2,'  End Right Sweep Memory')

    # Return result
    return E,EE,EEs,wgt

def left_sweep(mps,mpo,env,
               alg='davidson',
               start_site=None,
               end_site=None,
               noise=0.,
               orthonormalize=False,
               state_avg=True):
    """
    Perform a DMRG sweep from right to left

    Args: 
        mps : 1D Array of Matrix Product States
            The current mps list
            (stored as an mpsList
            as specified in cyclomps.tools.mps_tools)
        mpo : 1D Array
            An array that contains a list of mpos, each of which 
            is a list with an array for each site in the lattice
            representing the mpo at that site.
        env : 1D Array of an environment
            The environemtn for the current mps
            (stored as an envList
            as specified in cyclomps.tools.env_tools)

    Kwargs:
        alg : string
            The algorithm that will be used. Available options are
            'arnoldi', 'exact', and 'davidson', current default is 
            'davidson'.
            Default : 'davidson'
        start_site : int
            The site of the gauge for the input mps, where the DMRG
            sweep will start.
            Default : the right-most site
        end_site : int
            Where the DMRG sweep will end, 
            Default : the left-most site
        noise : float or 1D Array of floats
            The magnitude of the noise added to the mbd to prevent
            getting stuck in a local minimum
            !! NOT IMPLEMENTED !!
            Default : 0.
        orthonormalize : bool
            Specify whether to orthonormalize eigenvectors after solution
            of eigenproblem. This will cause problems for all systems
            unless the eigenstates being orthonormalized are degenerate.
            Default : False
        state_avg : bool
            Specify whether to use the state averaging procedure to target
            multiple states when doing the renormalization step.
            !! ONLY STATE AVG IS IMPLEMENTED !!
            Default : True

    Returns:
        E : 1D Array
            The energies for the number of states targeted
        EE : 1D List of floats
            The von neumann entanglement entropy for each state
        EEs : 1D List of 1D Array of floats
            The von neumann entanglement spectrum for each state
        wgt : 1D list of floats
            The discarded weight for each state
    """
    t0 = time.time()
    mpiprint(2,'\n')
    mpiprint(2,'Beginning Left Sweep')
    mpiprint(2,'-'*50)
    memprint(2,'  Start Left Sweep Memory')
    
    # Get info about mps
    nSite = len(mps[0])
    if start_site is None: start_site = nSite-1
    if end_site is None: end_site = 0

    # Variables to return
    E = None
    EE = None
    EEs = None
    wgt = None

    # Sweep through sites
    for site in range(start_site,end_site,-1):

        # Step left at each site
        E_,EE_,EEs_,wgt_ = left_step(site,mps,mpo,env,
                                     alg=alg,
                                     noise=noise,
                                     orthonormalize=orthonormalize,
                                     state_avg=state_avg)

        # Save results at center site
        if site == int(nSite/2):
            E = E_
            EE = EE_
            EEs = EEs_
            wgt = wgt_

    # Print total sweep time
    timeprint(3,'Left Sweep Time: {} s'.format(time.time()-t0))
    memprint(2,'  End Left Sweep Memory')

    # Return result
    return E,EE,EEs,wgt

def check_conv(cont,conv,niter,E,Eprev,tol,min_iter,max_iter):
    """
    Check for convergence during the DMRG sweeps
    """
    mpiprint(8,'Checking for convergence')
    
    # Check for convergence
    if not hasattr(E,'__len__'): E = nparray([E])
    if npsum(npabs((E-Eprev)/E)) < (tol*len(E)) and niter >= min_iter:
        conv = True
        cont = False
        mpiprint(1,'='*50)
        mpiprint(1,'Convergence acheived')
        mpiprint(1,'='*50)

    # Check if we have exceeded max iter
    elif niter >= max_iter:
        conv = False
        cont = False
        mpiprint(1,'!'*50)
        mpiprint(1,'Max Iteration Exceeded')
        mpiprint(1,'!'*50)

    # Increment niter and update Eprev
    niter += 1
    Eprev = E

    return cont,conv,niter,Eprev

def sweeps(mps,mpo,env,
           max_iter=5,
           min_iter=0,
           tol=1e-5,
           alg='davidson',
           noise=0.,
           orthonormalize=False,
           state_avg=True,
           start_gauge=0,
           end_gauge=0):
    """
    Run the sweeping in the one-site DMRG algorithm

    Args:
        mps : 1D Array of Matrix Product States
            The initial guess for the mps (stored as an mpsList
            as specified in cyclomps.tools.mps_tools)
        mpo : 1D Array
            An array that contains a list of mpos, each of which 
            is a list with an array for each site in the lattice
            representing the mpo at that site.
        env : 1D Array of an environment
            The initial environment for mps (stored as an envList
            as specified in cyclomps.tools.env_tools)

    Kwargs:
        tol : int or 1D Array of ints
            The relative convergence tolerance.             
            Default : 1.e-5
        max_iter : int or 1D Array of ints
            The maximum number of iterations
            Default : 10
        min_iter : int or 1D Array of ints
            The minimum number of iterations
            Default : 0
        noise : float or 1D Array of floats
            The magnitude of the noise added to prevent
            getting stuck in a local minimum
            !! NOT IMPLEMENTED !!
            Default : 0.
        alg : string
            The algorithm that will be used. Available options are
            'arnoldi', 'exact', and 'davidson', current default is 
            'davidson'.
            Default : 'davidson'
        orthonormalize : bool
            Specify whether to orthonormalize eigenvectors after solution
            of eigenproblem. This will cause problems for all systems
            unless the eigenstates being orthonormalized are degenerate.
            Default : False
        state_avg : bool
            Specify whether to use the state averaging procedure to target
            multiple states when doing the renormalization step.
            !! ONLY STATE AVG IS IMPLEMENTED !!
            Default : True
        start_gauge : int
            The initial site where the gauge is located.
            Default : 0
        end_gauge : int
            The site at which the gauge should be located when the 
            mps is returned.
            Default : 0

    Returns:
        E : 1D Array
            The energies for the number of states targeted
        EE : 1D List of floats
            The von neumann entanglement entropy for each state
        EEs : 1D List of 1D Array of floats
            The von neumann entanglement spectrum for each state
        wgt : 1D list of floats
            The discarded weight for each state
    """
    t0 = time.time()
    mpiprint(1,'\n\n')
    mpiprint(1,'Beginning DMRG Sweeping Algorithm')
    memprint(1,'  Starting Sweeps Memory')
    mpiprint(1,'='*50)

    # Get some useful info
    nSites = len(mps[0])

    # Set up parameters for sweeping
    start_site = start_gauge
    end_site = 0
    cont = True
    conv = False
    niter = 0
    Eprev = 0.
    final = False

    # Perform sweeps until convergence or max_iter
    while cont:
        
        # Run Right Sweep
        if niter == 0: 
            start_site = start_gauge
        else:
            start_site = 0

        res = right_sweep(mps,mpo,env,
                          alg=alg,
                          start_site=start_site,
                          end_site=nSites-1,
                          noise=noise,
                          orthonormalize=orthonormalize,
                          state_avg=state_avg)

        # Run Left Sweep
        res =  left_sweep(mps,mpo,env,
                          alg=alg,
                          start_site=nSites-1,
                          end_site=end_site,
                          noise=noise,
                          orthonormalize=orthonormalize,
                          state_avg=state_avg)

        # Check for Convergence
        E = res[0]
        cont,conv,niter,Eprev = check_conv(cont,conv,niter,E,Eprev,tol,min_iter,max_iter)

    # One last sweep, to move gauge
    resf= right_sweep(mps,mpo,env,
                      alg=alg,
                      start_site=0,
                      end_site=end_gauge,
                      noise=noise,
                      orthonormalize=orthonormalize,
                      state_avg=state_avg)

    # Print time for all sweeps
    timeprint(2,'Time for all sweeps: {} s'.format(time.time()-t0))
    memprint(1,'  Ending Sweeps Memory')

    # Return Results
    return res

def dmrg(mpo,
         mps=None, mpsl=None,
         env=None, envl=None,
         mbd=10, tol=1.e-5, max_iter=10, min_iter=0, noise=0.,
         mps_subdir='mps', env_subdir='env',
         mpsl_subdir='mpsl', envl_subdir='envl',
         nStates=2, dtype=complex_,
         fixed_bd=False, alg='davidson',
         return_state=False, return_env=False,
         return_entanglement=False, return_wgt=False,
         orthonormalize=False,
         state_avg=True, left=False, start_gauge=0, end_gauge=0):
    """
    Run the one-site DMRG algorithm

    Args:
        mpo : 1D Array
            An array that contains a list of mpos, each of which 
            is a list with an array for each site in the lattice
            representing the mpo at that site.

    Kwargs:
        mps : 1D Array of Matrix Product States
            The initial guess for the mps (stored as an mpsList
            as specified in cyclomps.tools.mps_tools)
            Default : None (mps0 will be random)
        mpsl : 1D Array of Matrix Product States
            The initial guess for the left mps (stored as an mpsList
            as specified in cyclomps.tools.mps_tools)
            Default : None (mps0 will be random)
        env : 1D Array of an environment
            The initial environment for mps (stored as an envList
            as specified in cyclomps.tools.env_tools)
            Default : None (env will be built in computation)
        envl : 1D Array of an environment
            The initial environment for the left mps (stored as an envList
            as specified in cyclomps.tools.env_tools)
            Default : None (env will be built in computation)
        mbd : int or 1D Array of ints
            The maximum bond dimension for the mps. 
            If this is a single int, then the bond dimension
            will be held constant for all sweeps. Otherwise, the 
            bond dimension will be incremented after max_iter sweeps
            or until the tolerance is reached.
            sweeps. (Note that if max_iter and/or tol is a list, we 
            require len(max_iter) == len(mbd) == len(tol), and the 
            maximum number of iterations or convergence tolerance
            changes with the retained maximum bond dimension.)
            Default : 10
        tol : int or 1D Array of ints
            The relative convergence tolerance. This may be a list, 
            meaning that as the mbd is increased, different tolerances
            are specified.
            Default : 1.e-5
        max_iter : int or 1D Array of ints
            The maximum number of iterations for each mbd
            Default : 10
        min_iter : int or 1D Array of ints
            The minimum number of iterations for each mbd
            Default : 0
        noise : float or 1D Array of floats
            The magnitude of the noise added to the mbd to prevent
            getting stuck in a local minimum
            !! NOT IMPLEMENTED !!
            Default : 0.
        mps_subdir : string
            The subdirectory under CALC_DIR (specified in cyclomps.tools.params)
            where the mps will be saved. 
            Default : 'mps'
        mpsl_subdir : string
            The subdirectory under CALC_DIR (specified in cyclomps.tools.params)
            where the left mps will be saved.
            Default : 'mpsl'
        env_subdir : string
            The subdirectory under CALC_DIR (specified in cyclomps.tools.params)
            where the environment will be saved.
            Default : 'env'
        envl_subdir : string
            The subdirectory under CALC_DIR (specified in cyclomps.tools.params)
            where the environment for the left mps will be saved.
            Default : 'envl'
        nStates : int
            The number of retained states
            Default : 2
        dtype : dtype
            The data type for the mps and env
            Default : np.complex_
        fixed_bd : bool
            This ensures that all bond dimensions are constant
            throughout the MPS, i.e. mps[0].dim = (1 x d[0] x mbd)
            instead of mps[0].dim = (1 x d[0] x d[0]), and so forth.
            Default : False
        alg : string
            The algorithm that will be used. Available options are
            'arnoldi', 'exact', and 'davidson', current default is 
            'davidson'.
            Default : 'davidson'
        return_state : bool
            Return the resulting mps list
            Default : False
        return_env : bool
            Return the resulting env list
            Default : False
        return_entanglement : bool
            Return the entanglement entropy and entanglement spectrum
            Default : False
        return_wgt : bool
            Return the discarded weights
            Default : False
        orthonormalize : bool
            Specify whether to orthonormalize eigenvectors after solution
            of eigenproblem. This will cause problems for all systems
            unless the eigenstates being orthonormalized are degenerate.
            Default : False
        state_avg : bool
            Specify whether to use the state averaging procedure to target
            multiple states when doing the renormalization step.
            Default : True
            !! ONLY STATE AVG IS IMPLEMENTED !!
        left : bool
            If True, then we calculate the left and right eigenstate
            otherwise, only the right.
            Default : False
        start_gauge : int
            The site at which the gauge should (or is) located in the 
            initial mps.
            Default : 0
        end_gauge : int
            The site at which the gauge should be located when the 
            mps is returned.
            Default : 0

    Returns:
        E : 1D Array
            The energies for the number of states targeted
        EE : 1D Array
            The entanglement entropy for the states targeted
            Returned only if return_entanglement == True
        EEs : 1D Array of 1D Arrays
            The entanglement entropy spectrum for the states targeted
            Returned only if return_entanglement == True
        mps : 1D Array of Matrix Product States
            The resulting matrix product state list
            Returned only if return_state == True
        env : 1D Array of an environment
            The resulting environment list
    """
    t0 = time.time()
    mpiprint(0,'\n\n')
    mpiprint(0,'Starting DMRG one-site calculation')
    memprint(1,'Initial Available Memory')
    mpiprint(0,'#'*50)

    # Check inputs for problems
    if not hasattr(mbd,'__len__'): mbd = nparray([mbd])
    if not hasattr(tol,'__len__'):
        tol = tol*npones(len(mbd))
    else:
        assert (len(mbd) == len(tol)), 'Lengths of mbd and tol do not agree'
    if not hasattr(max_iter,'__len__'):
        max_iter = max_iter*npones(len(mbd))
    else:
        assert (len(max_iter) == len(mbd)), 'Lengths of mbd and max_iter do not agree'
    if not hasattr(min_iter,'__len__'):
        min_iter = min_iter*npones(len(mbd))
    else:
        assert (len(min_iter) == len(mbd)), 'Lengths of mbd and min_iter do not agree'

    # --------------------------------------------------------------------------------
    # Solve for Right Eigenvector
    # --------------------------------------------------------------------------------
    # Determine local bond dimensions from mpo
    d = mpo_local_dim(mpo)

    # Create structures to save results
    return_E = npzeros((len(mbd),nStates))
    return_EE = npzeros((len(mbd),nStates))
    return_EEs = npzeros((len(mbd),nStates,max(mbd)))
    mps_res = []
    env_res = []
    # Loop through all maximum bond dimensions
    for mbd_ind, mbdi in enumerate(mbd):
        mpiprint(1,'\n')
        mpiprint(1,'/'*50)
        mpiprint(1,'Starting Calculation for mbd = {}'.format(mbdi))

        # Set up initial mps
        if mbd_ind == 0:
            if mps is None:
                # There is no previous guess to build on
                mps = create_mps_list(d,mbdi,nStates,
                                      dtype=dtype,fixed_bd=fixed_bd,
                                      subdir=mps_subdir+'_mbd'+str(mbdi)+'_')
                # Make sure it is in correct canonical form
                mps = make_mps_list_right(mps)
                mps = move_gauge(mps,0,start_gauge)
        else:
            # Increase the maximum bond dim of the previous system
            mps = increase_bond_dim(mps,mbdi,fixed_bd=fixed_bd)

        # Set up initial env
        if mbd_ind == 0:
            if env is None:
                env = calc_env(mps,mpo,
                               dtype=dtype,
                               subdir='env_mbd'+str(mbdi)+'_')
        else:
            env = calc_env(mps,mpo,
                           dtype=dtype,
                           gSite=end_gauge,
                           subdir='env_mbd'+str(mbdi)+'_')


        # Run the DMRG Sweeps
        outputr = sweeps(mps,mpo,env,
                         max_iter=max_iter[mbd_ind],
                         min_iter=min_iter[mbd_ind],
                         tol=tol[mbd_ind],
                         alg=alg,
                         noise=noise,
                         orthonormalize=orthonormalize,
                         state_avg=state_avg,
                         start_gauge=start_gauge,
                         end_gauge=end_gauge)

        # Collect results
        E = outputr[0]
        EE = outputr[1]
        EEs = outputr[2]
        wgt = outputr[3]
    # --------------------------------------------------------------------------------
    # Solve for Left Eigenvector
    # --------------------------------------------------------------------------------
    if left:
        mpiprint(0,'#'*50)
        mpiprint(0,'Left State')
        mpiprint(0,'#'*50)
        # Create left mpo
        mpol = mpo_conj_trans(mpo)        

        # Run this same function, but now with this new left mpo
        outputl = dmrg(mpo,
                       mps=mpsl, env=envl,
                       mbd=mbd, tol=tol, max_iter=max_iter, min_iter=min_iter, noise=noise,
                       mps_subdir=mpsl_subdir, env_subdir=envl_subdir,
                       nStates=nStates, dtype=dtype,
                       fixed_bd=fixed_bd, alg=alg,
                       return_state=True, return_env=True,
                       return_entanglement=True, return_wgt=True,orthonormalize=orthonormalize,
                       state_avg=state_avg, left=False, start_gauge=start_gauge, end_gauge=end_gauge)

        # Collect results
        El = outputl[0]
        EEl = outputl[1]
        EEls = outputl[2]
        wgtl = outputl[3]
        mpsl = outputl[4]
        envl = outputl[5]

    # ---------------------------------------------------------------------------------
    # Wrap Up Calculation
    # ---------------------------------------------------------------------------------
    # Print time for dmrg procedure
    timeprint(1,'Total time: {} s'.format(time.time()-t0))
    memprint(1,'Final Available Memory')

    # Return results
    ret_list = [E]
    if left: ret_list += [El]
    if return_entanglement:
        ret_list += [EE,EEs]
        if left: ret_list += [EEl,EEls]
    if return_wgt:
        ret_list += [wgt]
        if left: ret_list += [wgtl]
    if return_state:
        ret_list += [mps]
        if left: ret_list += [mpsl]
    if return_env:
        ret_list += [env]
        if left: ret_list += [envl]
    return ret_list
