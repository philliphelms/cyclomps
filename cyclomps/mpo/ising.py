from cyclomps.mpo.ops import *
from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
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
        gen_mpo = array([[I,   z, z],
                         [X,   z, z],
                         [h*Z, X, I]])
        if (site == 0):
            mpo[site] = expand_dims(gen_mpo[2,:],0)
        elif (site == N-1):
            mpo[site] = expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
        print(mpo[site])
    mpoL.append(mpo)
    mpoL = reorder_bonds(mpoL)
    return mpoL
