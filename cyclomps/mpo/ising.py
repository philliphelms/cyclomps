from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
from cyclomps.mpo.ops import *
import collections

############################################################################
# Ising Model
#
# A very simple implementation of the ising model mpo
###########################################################################

def return_mpo(N,hamParams):
    h = hamParams[0]
    print('h = {}'.format(h))
    # List to hold all mpos
    mpoL = []
    # Mainmpo
    mpo = [None]*N
    for site in range(N):
        if (site == 0):
            mpo[site] = array([[h*Z, X, I]])
        elif (site == N-1):
            mpo[site] = array([[I],[X],[h*Z]])
        else:
            mpo[site] = array([[I,   z, z],
                               [X,   z, z],
                               [h*Z, X, I]])
    # Add mpo to mpo list
    mpoL.append(mpo)
    # Reorder the bonds
    mpoL = reorder_bonds(mpoL)
    # Return results
    return mpoL
