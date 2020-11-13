from numpy import float_
from cyclomps.mpo.ops import *
from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
import collections
from numpy import exp,float_

############################################################################
# Rydberg Atoms
# Using Compression algorithm for 
# Long range interaction via 
# https://arxiv.org/pdf/1702.03650.pdf
###########################################################################

def return_mpo(N,hamParams,D=100,const=0.):
    # Extract Parameter Values
    Omega = hamParams[0]
    Delta = hamParams[1]
    V     = hamParams[2]

    # Make Interaction Matrix V_{n,n'} (Eq. B1 & B4
    Vnn_ = zeros((N,N),dtype=float_)
    for n1 in range(N):
        for n2 in range(n1+1,N):
            Vnn_[n1,n2] = V*(n2-n1)**(-6)

    # Get upper blocks
    Vp = []
    for p in range(N):
        Vp.append(Vnn_[0:p+1,p:])

    if True:
        # Explicit SVDs ##################################################
        Up = []
        Sp = []
        Wp = []
        for p in range(N):
            U,S,W = svd(Vp[p])
            # Do Truncation
            if len(S) > D:
                U = U[:,:D]
                S = S[:D]
                W = W[:D,:]
            # Save results
            Up.append(U)
            Sp.append(S)
            Wp.append(W)
        
        # Now Calculate X tensors
        Xp = []
        Xp.append(Up[0])
        for p in range(1,N):
            Upm1 = add_row_col(Up[p-1])
            _Upm1= pinv(Upm1)
            X = einsum('ij,jk->ik',_Upm1,Up[p])
            Xp.append(X)

        # Now Calculate Omega tensors
        Op = []
        for p in range(N):
            Op.append(einsum('ij,j,jk->ik',Xp[p],Sp[p],Wp[p]))

    else:
        # Recursive SVDs ##################################################
        Up = []
        Sp = []
        Wp = []
        Xp = []
        Op = []

        # Do SVD of Vp[0]
        U,S,W = svd(Vp[0])
        Up.append(U)
        Xp.append(U)
        Sp.append(S)
        Wp.append(W)
        Op.append(einsum('ij,j,jk->ik',Xp[0],Sp[0],Wp[0]))
        
        # Now move on to second (and remaining) site(s)
        for p in range(1,N):
            # Calculate Omega
            SW = einsum('i,ij->ij',S,W[:,1:])
            (n1,n2) = Vp[p].shape
            O = add_row(SW,Vp[p][n1-1,:])
            X,S,W = svd(O)
            # Do Truncation
            if len(S) > D:
                X = X[:,:D]
                S = S[:D]
                W = W[:D,:]
            U = einsum('ij,jk->ik',add_row_col(U),X)
            Up.append(U)
            Xp.append(X)
            Sp.append(S)
            Wp.append(W)
            Op.append(O)
    
    # Put compressed operators into an MPO
    mpo = []
    for site in range(N):

        # Create the generic local operator
        local_op = -Omega*Sx - Delta*v

        if site == 0:
            # Create the MPO Tensor
            local_op += const*I
            ten = array([[local_op, Xp[0][0,0]*v, I]])
            mpo.append(-ten)
        elif site == N-1:
            # Create an empty tensor
            (n1,n2) = Xp[site].shape
            ten_size = (n1+1,1,2,2)
            ten = zeros(ten_size,dtype=float_)
            # Insert left column
            for om_iter in range(n1-1):
                ten[om_iter+1,0,:,:] = Op[site][om_iter,0]*v
            # Insert identity at top
            ten[0,0,:,:] = I
            # Insert local op at bottom
            ten[n1,0,:,:] = local_op
            mpo.append(ten)
        else:
            # Create an empty tensor
            (n1,n2) = Xp[site].shape
            ten_size = (n1+1,n2+2,2,2)
            ten = zeros(ten_size,dtype=float_)

            # Insert identities
            ten[0,0,:,:] = I
            ten[n1,n2+1,:,:] = I

            # Insert left column
            for om_iter in range(n1-1):
                ten[om_iter+1,0,:,:] = Op[site][om_iter,0]*v # PH - Might be swapped?

            # Insert central X tensors
            (n1,n2) = Xp[site].shape 
            for xi1 in range(n1):
                for xi2 in range(n2):
                    if xi1 == n1-1:
                        ten[xi1+1,xi2+1,:,:] = Xp[site][xi1,xi2]*v # PH - Might be swapped again...
                    else:
                        ten[xi1+1,xi2+1,:,:] = Xp[site][xi1,xi2]*I # PH - Might be swapped again...

            # Insert local operators
            ten[ten_size[0]-1,0,:,:] = local_op

            # Put the tensor into the MPO
            mpo.append(ten)
    # Reshape MPO
    mpo = [mpo]
    mpo = reorder_bonds(mpo)
    return mpo

def add_row_col(Up):
    """
    Add a bottom row and right column of zeros to U, except for a 
    1 in the bottom right position.
    """
    U = zeros((Up.shape[0]+1,Up.shape[1]+1),dtype=float_)
    U[Up.shape[0],Up.shape[1]] = 1.
    U[:Up.shape[0],:Up.shape[1]] = Up
    return U

def add_row(SW,r):
    """
    Add a row to the SW matrix
    """
    (n1,n2) = SW.shape
    SWr = zeros((n1+1,n2),dtype=SW.dtype)
    SWr[:n1,:] = SW
    SWr[n1,:] = r
    return SWr
