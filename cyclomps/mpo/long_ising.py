from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
from cyclomps.mpo.ops import *
import collections

############################################################################
# Ising Model with transverse (hz) and longitudinal (hx) magnetic field
#
# A very simple implementation of the ising model mpo
###########################################################################

def return_mpo(N,hamParams,periodic=True):
    # Extract Hamiltonian Parameters
    (J,hx,hz) = hamParams

    # List to hold all mpos
    mpoL = []

    # Mainmpo
    mpo = [None]*N
    for site in range(N):
        if (site == 0):
            mpo[site] = array([[-J*hx*X-J*hz*Z, -J*X, I]])
        elif (site == N-1):
            mpo[site] = array([[I],[X],[-J*hx*X-J*hz*Z]])
        else:
            mpo[site] = array([[I,                 z, z],
                               [X,                 z, z],
                               [-J*hx*X-J*hz*Z, -J*X, I]])
    
    # Add mpo to mpo list
    mpoL.append(mpo)
    
    # Add periodic interaction if needed 
    mpo = [None]*N
    mpo[0] = array([[-J*X]])
    mpo[-1] = array([[X]])
    mpoL.append(mpo)
    
    # Reorder the bonds
    mpoL = reorder_bonds(mpoL)
    
    # Return results
    return mpoL
