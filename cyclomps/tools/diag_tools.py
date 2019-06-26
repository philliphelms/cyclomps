"""
Tools for Diagonalization in MPS algorithms

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

.. To Do:

"""

import scipy.linalg as sla
from cyclomps.tools.utils import *
from cyclomps.tools.params import *
from cyclomps.tools.mps_tools import mps_load_ten
from cyclomps.tools.env_tools import env_load_ten
from pyscf.lib import eig as davidson
from scipy.sparse.linalg import eigs as arnoldi
from scipy.sparse.linalg import LinearOperator
from numpy import complex_
from numpy import argsort as npargsort
from numpy import real as npreal
from numpy import array as nparray
from numpy import eye as npeye
from numpy import zeros as npzeros

def davidson1(mps,mpo,envl,envr):
    """ 
    Calculate the eigenvalues and eigenvectors with the davidson algorithm

    args:
        mps : 1d array of np or ctf tensors
            a list containing the mps tensor for each desired 
            state at the optimization site
        mpo : 1d array of np or ctf tensors
            a list containing the mpo tensor for each 
            operator at the optimization site
        envl : 1d array of np or ctf tensors
            a list containing the left env tensor for each
            operator at the optimization site
        envr : 1d array of np or ctf tensors
            a list containing the right env tensor for each
            operator at the optimization site

    returns:
        e : 1d array
            a 1d array of the energy associated with each state 
            of the system
        mps : 1d array of np or ctf tensors
            a list containing the resulting mps tensor for each
            state from the optimization
        ovlp : float
            the overlap between the input guess and output state
    """
    mpiprint(6,'Doing Davidson optimization routine')

    # Compute the number of states
    nStates = len(mps)

    # Make the hamiltonian function
    hop,precond = make_ham_func1(mps,mpo,envl,envr)

    # Determine initial guess
    guess = []
    for state in range(nStates):
        if USE_CTF: 
            guess.append(to_nparray(ravel(mps[state])))
        else:
            guess.append(ravel(mps[state]))

    # Send to davidson algorithm
    E,vecso = davidson(hop,guess,precond,
                         nroots=nStates,pick=pick_eigs,
                         follow_state=False,tol=DAVIDSON_TOL,
                         max_cycle=DAVIDSON_MAX_ITER)
    E = -E 
    # Sort results
    inds = npargsort(npreal(E))[::-1]
    # davidson occasionally does not return enough states
    if hasattr(E,'__len__'):
        E = E[inds[:nStates]]
        vecs = npzeros((len(vecso[0]),nStates),dtype=type(vecso[0][0]))
        for vec_ind in range(min(nStates,len(inds))):
            vecs[:,vec_ind] = vecso[inds[vec_ind]]
    else:
        vecs = vecso

    # Convert vecs back to ctf if needed
    if USE_CTF: vecs = from_nparray(vecs)

    # Check the overlap
    # PH - Need to implement
    ovlp = None

    # Convert vecs into original mps shape
    mps = vec2mps(vecs,mps)
    return E,mps,ovlp

def pick_eigs(w,v,nroots,x0):
    """
    Used in Davidson function to pick eigenvalues
    """
    mpiprint(10,'Picking Eigs in davidson or arnoldi')
    idx = npargsort(npreal(w))
    w = w[idx]
    v = v[:,idx]
    return w,v,idx

def make_ham_func1(mps,mpo,envl,envr):
    """
    Make hamiltonian function and preconditioner for implicit eigensolvers

    Args:
        mps : 1d array of np or ctf tensors
            a list containing the mps tensor for each desired 
            state at the optimization site
        mpo : 1d array of np or ctf tensors
            a list containing the mpo tensor for each 
            operator at the optimization site
        envl : 1d array of np or ctf tensors
            a list containing the left env tensor for each
            operator at the optimization site
        envr : 1d array of np or ctf tensors
            a list containing the right env tensor for each
            operator at the optimization site

    Returns:
        hop : Function
            The function f(x) = H*x
        precond : function
            A function to produce the preconditioner for the 
            davidson algorithm
    """
    mpiprint(8,'Creating Hamiltonian Functions for eigensolver')

    # Create hamiltonian function
    def hop(x):
        memprint(9,'Inside hop function')
        mpiprint(9,'Inside hop function')
        # Process input
        if USE_CTF: x = from_nparray(x)
        x = reshape(x,mps[0].shape)
        res = zeros_like(x,dtype=complex_)
        # Loop over all operators
        for op in range(len(mpo)):
            if mpo[op] is None:
                tmp = einsum('Bcb,apb->Bcap',envr[op],x)
                res +=einsum('Ada,Bcap->ApB',envl[op],tmp)
            else:
                tmp = einsum('Bcb,apb->Bcap',envr[op],x)
                tmp = einsum('dqpc,Bcap->dqBa',mpo[op],tmp)
                res +=einsum('Ada,dqBa->AqB',envl[op],tmp)
        res = ravel(res)
        if USE_CTF: res = to_nparray(res)
        return -res

    # Create Preconditioner Function
    if USE_PRECOND:
        diag = ravel(calc_diag1(mpo,envl,envr))
        if USE_CTF: diag = to_nparray(diag)
        def precond(dx,e,x0):
            return dx/(diag-e)
    else:
        def precond(dx,e,x0):
            return dx

    # Return resulting functions
    return hop,precond

def calc_diag1(mpo,envl,envr):
    """
    Calc the diagonal elements of the hamiltonian

    Args:
        mpo : 1d array of np or ctf tensors
            a list containing the mpo tensor for each 
            operator at the optimization site
        envl : 1d array of np or ctf tensors
            a list containing the left env tensor for each
            operator at the optimization site
        envr : 1d array of np or ctf tensors
            a list containing the right env tensor for each
            operator at the optimization site

    Returns:
        diag : 1D ctf or np array
            The diagonal of the local hamiltonian defined by 
            the input mpo and environment tensors.
    """
    mpiprint(7,'Calculating diagonal elements of ham for precond')

    # Get number of operators
    nOps = len(mpo)

    # Loop through operators to compute diagonal
    for op in range(nOps):
        if mpo[op] is None:
            identity = array(nparray([[npeye(2)]]))
            identity = einsum('abpq->apqb',identity)
            tmp_op = identity
        else:
            tmp_op = mpo[op]
        diag_mpo = einsum('cppd->cpd',tmp_op)
        diag_envl= einsum('aca->ac',envl[op])
        diag_envr= einsum('bdb->bd',envr[op])
        tmp = einsum('ac,cpd->apd',diag_envl,diag_mpo)
        if op == 0:
            diag = einsum('apd,bd->apb',tmp,diag_envr)
        else:
            diag += einsum('apd,bd->apb',tmp,diag_envr)

    return diag
    
    
def arnoldi1(mps,mpo,envl,envr):
    """ 
    Calculate the eigenvalues and eigenvectors with the arnoldi algorithm
    
    Args:
        mps : 1d array of np or ctf tensors
            a list containing the mps tensor for each desired 
            state at the optimization site
        mpo : 1d array of np or ctf tensors
            a list containing the mpo tensor for each 
            operator at the optimization site
        envl : 1d array of np or ctf tensors
            a list containing the left env tensor for each
            operator at the optimization site
        envr : 1d array of np or ctf tensors
            a list containing the right env tensor for each
            operator at the optimization site

    Returns:
        E : 1d array
            a 1d array of the energy associated with each state 
            of the system
        mps : 1d array of np or ctf tensors
            a list containing the resulting mps tensor for each
            state from the optimization
        ovlp : float
            the overlap between the input guess and output state
    """
    mpiprint(6,'Doing Arnoldi optimization routine')

    # Compute the number of states
    nStates = len(mps)
    (n1,n2,n3) = mps[0].shape

    # Make the hamiltonian function
    hop,_ = make_ham_func1(mps,mpo,envl,envr)
    hop = LinearOperator((n1*n2*n3,n1*n2*n3),matvec=hop)

    # Determine initial guess
    if USE_CTF:
        guess = to_nparray(ravel(mps[0]))
    else:
        guess = ravel(mps[0])
    #guess = np.zeros((n1*n2*n3,nStates),dtype=type(mps[0][0,0,0]))
    #for state in range(nStates):
    #    if USE_CTF:
    #        guess[:,state] = to_nparray(ravel(mps[state]))
    #    else:
    #        guess[:,state] = ravel(mps[state])

    # Send to davidson algorithm
    try:
        E,vecs = arnoldi(hop,k=nStates,which='SR',
                            v0=guess,tol=ARNOLDI_TOL,
                            maxiter=ARNOLDI_MAX_ITER)
    except Exception as exc:
        E = exc.eigenvalues
        vecs = exc.eigenvectors
    E = -E 

    # Sort results
    inds = npargsort(E)[::-1]
    E = E[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]

    # Convert vecs back to ctf if needed
    if USE_CTF: vecs = from_nparray(vecs)

    # Check the overlap
    # PH - Need to implement
    ovlp = None

    # Convert vecs into original mps shape
    mps = vec2mps(vecs,mps)
    return E,mps,ovlp


def exact1(mps,mpo,envl,envr):
    """ 
    Calculate the eigenvalues and eigenvectors by explicitly computing the Hamiltonian

    Args:
        mps : 1d array of np or ctf tensors
            a list containing the mps tensor for each desired 
            state at the optimization site
        mpo : 1d array of np or ctf tensors
            a list containing the mpo tensor for each 
            operator at the optimization site
        envl : 1d array of np or ctf tensors
            a list containing the left env tensor for each
            operator at the optimization site
        envr : 1d array of np or ctf tensors
            a list containing the right env tensor for each
            operator at the optimization site

    Returns:
        E : 1d array
            a 1d array of the energy associated with each state 
            of the system
        mps : 1d array of np or ctf tensors
            a list containing the resulting mps tensor for each
            state from the optimization
        ovlp : float
            the overlap between the input guess and output state
    """
    mpiprint(6,'Doing Exact optimization routine')
    
    # Figure out number of states required
    nState = len(mps)

    # Compute the full Hamiltonian
    H = calc_ham1(mps,mpo,envl,envr)

    # Diagonalize and sort
    if USE_CTF: H = to_nparray(H)
    E,vecs = sla.eig(H)
    inds = npargsort(E)[::-1]
    E = E[inds[:nState]]
    vecs = vecs[:,inds[:nState]]

    # Convert vecs back to ctf if needed
    if USE_CTF: vecs = from_nparray(vecs)

    # Check the overlap
    # PH - Need to implement
    ovlp = None

    # Convert vecs into original mps shape
    mps = vec2mps(vecs,mps)

    return E,mps,ovlp

def vec2mps(vecs,mps):
    """
    Put a vector into an mps
    """
    mpiprint(8,'Putting vector into mps form')
    
    # Determine number of states in mps
    nStates = len(mps)
    if len(vecs.shape) == 1:
        nStatesVec = 1
        mps[0] = reshape(vecs,mps[0].shape) # PH - Might need some swapping of indices here?
    else:
        (_,nStatesVec) = vecs.shape
        for state in range(nStates):
            mps[state] = reshape(vecs[:,state],mps[state].shape) # PH - Might need some swapping of indices here?
    return mps

def calc_ham1(mps,mpo,envl,envr,dtype=None):
    """
    Compute the full one-site local hamiltonian

    Args:
        mps : 1d array of np or ctf tensors
            a list containing the mps tensor for each desired 
            state at the optimization site
        mpo : 1d array of np or ctf tensors
            a list containing the mpo tensor for each 
            operator at the optimization site
        envl : 1d array of np or ctf tensors
            a list containing the left env tensor for each
            operator at the optimization site
        envr : 1d array of np or ctf tensors
            a list containing the right env tensor for each
            operator at the optimization site

    Returns:
        H : 2d np or ctf array
            The Resulting Hamiltonian
    """
    # Determine Hamiltonian Dimensions
    (n1,n2,n3) = mps[0].shape
    dim = n1*n2*n3

    # Allocate Memory
    if dtype is None: dtype = type(envl[0][0,0,0])
    H = zeros((dim,dim),dtype=dtype)

    # Populate Hamiltonian
    nOps = len(mpo)
    for op in range(nOps):
        if mpo[op] is None:
            # Replace None operator with identity
            I = array(nparray([[npeye(n2)]]))
            I = einsum('abpq->apqb',I)
            Hop = einsum('cqpd,Ada->Aqcpa',I,envr[op])
        else:
            Hop = einsum('cqpd,Ada->Aqcpa',mpo[op],envr[op])
        Hop = einsum('Aqcpa,Bcb->BqAbpa',Hop,envl[op]) # PH - Check Hamiltonian Ordering
        H += reshape(Hop,(dim,dim))

    # Return resulting hamiltonian
    return H

def eig1(mps,mpo,envl,envr,
         alg='davidson'):
    """
    Calculate the eigenvalues and vectors for an MPS at a given site(s)

    Args:
        mps : 1D array of np or ctf tensors
            A list containing the mps tensor for each desired 
            state at the optimization site
        mpo : 1D Array of np or ctf tensors
            A list containing the mpo tensor for each 
            operator at the optimization site
        envl : 1D Array of np or ctf tensors
            A list containing the left env tensor for each
            operator at the optimization site
        envr : 1D Array of np or ctf tensors
            A list containing the right env tensor for each
            operator at the optimization site

    Kwargs:
        alg : string
            The algorithm that will be used. Available options are
            'arnoldi', 'exact', and 'davidson', current default is 
            'davidson'.

    Returns:
        E : 1D Array
            A 1D array of the energy associated with each state 
            of the system
        mps : 1D Array of np or ctf tensors
            A list containing the resulting mps tensor for each
            state from the optimization
        ovlp : float
            The overlap between the input guess and output state
    """
    if alg == 'davidson':
        E,mps,ovlp = davidson1(mps,mpo,envl,envr)
        
    elif alg == 'exact':
        E,mps,ovlp = exact1(mps,mpo,envl,envr)

    elif alg == 'arnoldi':
        E,mps,ovlp = arnoldi1(mps,mpo,envl,envr)

    return E,mps,ovlp
