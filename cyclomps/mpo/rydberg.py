from numpy import float_
from cyclomps.mpo.ops import *
from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
import collections
from numpy import exp

############################################################################
# Rydberg Atoms
###########################################################################

def return_mpo(N,hamParams):
    # Extract Parameter Values
    Omega = hamParams[0]
    Delta = hamParams[1]
    V     = hamParams[2]

    # Create the main mpo
    mpo = []
    for site in range(N):

        # Create the generic local operator
        local_op = -Omega*Sx - Delta*n
        
        # First Site (row mpo)
        if (site == 0):
            # Create the MPO tensor
            ten = array([[local_op, n, I]])
            mpo.append(-ten)

        # Last site (column mpo)
        elif (site == N-1):
            # Creat the mpo tensor
            ten = zeros((N+1,1,2,2),dtype=float_)

            # Fill in column
            ten[0,0,:,:] = I
            for site2 in range(N-1):
                coef = (N-site2-1)**(-6.)
                ten[site2+1,0,:,:] = coef*n
            ten[N,0,:,:] = local_op
            mpo.append(ten)

        # Interior Sites (full mpo)
        else:
            # Create the mpo tensor
            ten = zeros((site+2,site+3,2,2),dtype=float_)

            # Fill in first column
            ten[0,0,:,:] = I
            for site2 in range(site):
                coef = (site-site2)**(-6.)
                ten[site2+1,0,:,:] = coef*n
            ten[site+1,0,:,:] = local_op

            # Fill in bottom row
            ten[site+1,site+1] = n
            ten[site+1,site+2] = I

            # Fill in interior with identities
            for site2 in range(site):
                ten[site2+1,site2+1] = I

            mpo.append(ten)

    mpo = [mpo]
    mpo = reorder_bonds(mpo)
    return mpo
