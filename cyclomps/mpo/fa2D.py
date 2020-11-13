from numpy import float_
from cyclomps.mpo.ops import *
from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
import collections
from numpy import exp

############################################################################
# 2D FA model
###########################################################################

def return_mpo(N,hamParams,hermitian=False):
    if hasattr(N,'__len__'):
        Nx = N[0]
        Ny = N[1]
    else:
        Nx = N
        Ny = N
    # Generate MPO
    mpo = open_mpo(Nx,Ny,hamParams,hermitian=hermitian)
    # Reorder bonds to match mps 
    mpo = reorder_bonds(mpo)
    return mpo

def open_mpo(Nx,Ny,hamParams,hermitian=False):
    # Extract parameter Values
    (c,s) = hamParams
    es = exp(-s)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 6+(Ny-2)*2
    mpiprint(5,'Hamiltonian Bond Dimension = {}'.format(ham_dim))
    for xi in range(Nx):
        for yi in range(Ny):
            # Figure out hopping term
            if hermitian:
                flipOp = (es*sqrt(c*(1.-c))*X - c*v - (1.-c)*n)
            else:
                flipOp = (c*es*Sp + (1-c)*es*Sm - c*v - (1.-c)*n)
            # Build generic MPO
            # Fill left column
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = flipOp
            gen_mpo[Ny,0,:,:] = flipOp
            gen_mpo[Ny+1,0,:,:] = n
            gen_mpo[2*Ny,0,:,:] = n
            # Fill interior
            col_ind = 1
            row_ind = 2
            for k in range(2):
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build Bottom Row
            gen_mpo[ham_dim-1,Ny,:,:] = n
            gen_mpo[ham_dim-1,2*Ny,:,:] = flipOp
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

##########################################################################
# Useful Functions
##########################################################################

def val2matParams(Nx,Ny,hamParams):
    cx = hamParams[0]
    cy = hamParams[0]
    sx = hamParams[0]
    sy = hamParams[0]
    # Set interior rates
    cx_m = cx*ones((Nx,Ny),dtype=float_)
    cy_m = cy*ones((Nx,Ny),dtype=float_)
    # Set bias
    sx_m = sx*ones((Nx,Ny),dtype=float_)
    sy_m = sy*ones((Nx,Ny),dtype=float_)
    return (cx_m,cy_m,sx_m,sy_m)

def extractParams(hamParams):
    cx = hamParams[0].astype(dtype=float_)
    cy = hamParams[0].astype(dtype=float_)
    sx = hamParams[0].astype(dtype=float_)
    sy = hamParams[0].astype(dtype=float_)
    return (cx,cy,sx,sy)
