import ctf
import numpy as np
from pyscf.lib import eig
import time
from cyclomps.tools.utils import mpiprint
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

def main():
    t0 = time.time()

    ######## Inputs ##############################
    # SEP Model
    N = 50
    alpha = 0.35 # In at left
    beta = 2./3. # Exit at right
    s = -1.      # Exponential weighting
    gamma = 0.   # Exit at left
    delta = 0.   # In at right
    p = 1.       # Jump right
    q = 0.       # Jump Left
    # Optimization
    tol = 1e-5
    maxIter = 0
    maxBondDim = 10
    useCTF = True
    ##############################################

    # Create MPS #################################
    # PH - Make Isometries, Center Site
    mpiprint('Generating MPS')
    M = []
    for i in range(int(N/2)):
        tmp = np.random.rand(2,
                             min(2**(i),maxBondDim),
                             min(2**(i+1),maxBondDim))\
                             +0.j
        M.append(ctf.from_nparray(tmp))
    for i in range(int(N/2))[::-1]:
        tmp = np.random.rand(2,
                             min(2**(i+1),maxBondDim),
                             min(2**i,maxBondDim))\
                             +0.j
        M.append(ctf.from_nparray(tmp))
    ##############################################

    # Create MPO #################################
    mpiprint('Generating MPO')
    # Simple Operators
    Sp = np.array([[0.,1.],[0.,0.]])
    Sm = np.array([[0.,0.],[1.,0.]])
    n = np.array([[0.,0.],[0.,1.]])
    v = np.array([[1.,0.],[0.,0.]])
    I = np.array([[1.,0.],[0.,1.]])
    z = np.array([[0.,0.],[0.,0.]])
    # List to hold MPOs
    W = []
    # First Site
    site_0 = ctf.astensor([[alpha*(np.exp(-s)*Sm-v),
                            np.exp(-s)*Sp,
                            -n,
                            I                       ]])
    W.append(site_0)
    # Central Sites
    for i in range(N-2):
        site_i = ctf.astensor([[I,             z,  z, z],
                               [Sm,            z,  z, z],
                               [v,             z,  z, z],
                               [z, np.exp(-s)*Sp, -n, I]])
        W.append(site_i)
    # Last Site
    site_N = ctf.astensor([[I                     ],
                           [Sm                    ],
                           [v                     ],
                           [beta*(np.exp(-s)*Sp-n)]])
    W.append(site_N)
    ##############################################

    # Canonicalize MPS ###########################
    for i in range(int(N)-1,0,-1):
        M_reshape = ctf.transpose(M[i],axes=[1,0,2])
        (n1,n2,n3) = M_reshape.shape
        M_reshape = M_reshape.reshape(n1,n2*n3)
        (U,S,V) = ctf.svd(M_reshape)
        M_reshape = V.reshape(n1,n2,n3)
        M[i] = ctf.transpose(M_reshape,axes=[1,0,2])
        M[i-1] = ctf.einsum('klj,ji,i->kli',M[i-1],U,S)
    ##############################################

    # Canonicalize MPS ###########################
    def pick_eigs(w,v,nroots,x0):
        idx = np.argsort(np.real(w))
        w = w[idx]
        v = v[:,idx]
        return w, v, idx
    ##############################################

    # Create Environment #########################
    mpiprint('Generating Environment')
    # Allocate empty environment
    F = []
    tmp = np.array([[[1.]]])+0.j
    F.append(ctf.from_nparray(tmp))
    for i in range(int(N/2)):
        tmp = np.zeros((min(2**(i+1),maxBondDim),4,min(2**(i+1),maxBondDim)))+0.j
        F.append(ctf.from_nparray(tmp))
    for i in range(int(N/2)-1,0,-1):
        tmp = np.zeros((min(2**(i),maxBondDim),4,min(2**i,maxBondDim)))+0.j
        F.append(ctf.from_nparray(tmp))
    tmp = np.array([[[1.]]])+0.j
    F.append(ctf.from_nparray(tmp))
    # Calculate initial environment
    for i in range(int(N)-1,0,-1):
        tmp = ctf.einsum('eaf,cdf->eacd',M[i],F[i+1])
        tmp = ctf.einsum('ydbe,eacd->ybac',W[i],tmp)
        F[i] = ctf.einsum('bxc,ybac->xya',ctf.conj(M[i]),tmp)
    ##############################################

    # Optimization Sweeps ########################
    converged = False
    iterCnt = 0
    E_prev = 0
    while not converged:
        # Right Sweep ----------------------------
        tr = time.time()
        mpiprint('Start Right Sweep {}'.format(iterCnt))
        for i in range(N-1):
            (n1,n2,n3) = M[i].shape
            # Make Hx Function
            def Hfun(x):
                x_reshape = ctf.array(x)
                x_reshape = ctf.reshape(x_reshape,(n1,n2,n3))
                tmp = ctf.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
                tmp = ctf.einsum('njol,ijlm->noim',W[i],tmp)
                res = ctf.einsum('pnm,noim->opi',F[i],tmp)
                return -ctf.reshape(res,-1).to_nparray()
            def precond(dx,e,x0):
                return dx
            # Set up initial guess
            guess = ctf.reshape(M[i],-1).to_nparray()
            # Run eigenproblem
            u,v = eig(Hfun,guess,precond,pick=pick_eigs)
            E = -u
            v = ctf.array(v)
            M[i] = ctf.reshape(v,(n1,n2,n3))
            # Print Results
            mpiprint('\tEnergy at site {} = {}'.format(i,E))
            # Right Normalize
            M_reshape = ctf.reshape(M[i],(n1*n2,n3))
            (U,S,V) = ctf.svd(M_reshape)
            M[i] = ctf.reshape(U,(n1,n2,n3))
            M[i+1] = ctf.einsum('i,ij,kjl->kil',S,V,M[i+1])
            # Update F
            tmp = ctf.einsum('jlp,ijk->lpik',F[i],ctf.conj(M[i]))
            tmp = ctf.einsum('lmin,lpik->mpnk',W[i],tmp)
            F[i+1] = ctf.einsum('npq,mpnk->kmq',M[i],tmp)
        mpiprint('Complete Right Sweep {}, {} sec'.format(iterCnt,time.time()-tr))
        # Left Sweep ------------------------------
        tl = time.time()
        mpiprint('Start Left Sweep {}'.format(iterCnt))
        for i in range(N-1,0,-1):
            (n1,n2,n3) = M[i].shape
            # Make Hx Function
            def Hfun(x):
                x_reshape = ctf.array(x)
                x_reshape = ctf.reshape(x_reshape,(n1,n2,n3))
                tmp = ctf.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
                tmp = ctf.einsum('njol,ijlm->noim',W[i],tmp)
                res = ctf.einsum('pnm,noim->opi',F[i],tmp)
                return -ctf.reshape(res,-1).to_nparray()
            def precond(dx,e,x0):
                return dx
            # Set up initial guess
            guess = ctf.reshape(M[i],-1).to_nparray()
            # Run eigenproblem
            u,v = eig(Hfun,guess,precond,pick=pick_eigs)
            E = -u
            v = ctf.array(v)
            M[i] = ctf.reshape(v,(n1,n2,n3))
            # Print Results
            mpiprint('\tEnergy at site {}= {}'.format(i,E))
            # Right Normalize 
            M_reshape = ctf.transpose(M[i],(1,0,2))
            M_reshape = ctf.reshape(M_reshape,(n2,n1*n3))
            (U,S,V) = ctf.svd(M_reshape)
            M_reshape = ctf.reshape(V,(n2,n1,n3))
            M[i] = ctf.transpose(M_reshape,(1,0,2))
            M[i-1] = ctf.einsum('klj,ji,i->kli',M[i-1],U,S)
            # Update F
            tmp = ctf.einsum('eaf,cdf->eacd',M[i],F[i+1])
            tmp = ctf.einsum('ydbe,eacd->ybac',W[i],tmp)
            F[i] = ctf.einsum('bxc,ybac->xya',ctf.conj(M[i]),tmp)
        mpiprint('Left Sweep {}, {} sec'.format(iterCnt,time.time()-tl))
        # Convergence Test -----------------------
        if np.abs(E-E_prev) < tol:
            mpiprint('#'*75+'\nConverged at E = {}'.format(E)+'\n'+'#'*75)
            converged = True
        elif iterCnt > maxIter:
            mpiprint('Convergence not acheived')
            converged = True
        else:
            iterCnt += 1
            E_prev = E
    mpiprint('Total Time = {}'.format(time.time()-t0))

if __name__ == "__main__":
    main()
