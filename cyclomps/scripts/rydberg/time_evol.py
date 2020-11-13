from cyclomps.mpo.rydberg_compressed import return_mpo
from cyclomps.tools.mpo_tools import mpo_local_dim
from cyclomps.tools.mps_tools import create_simple_state
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.algs.tddmrg import tddmrg
from cyclomps.tools.utils import *

# System Size
N = 10
mbd = 50
Dmpo = 3

# Time evolution parameters
dt = 0.001
tf = 1.

# Initial Hamiltonian Parameters
Omega_0 = 1. 
Delta_0 = 0.5
V_0     = 1.

# Final Hamiltonian Parameters
Omega_f = 1.e-3
Delta_f = 0.5
V_f     = 1.

# Get the MPO
hamParams = (Omega_0,Delta_0,V_0)
mpo = return_mpo(N,hamParams,D=Dmpo)

# Create an initial state
energy_gs,mps = dmrg(mpo,
                     mbd=10,
                     left=False,
                     nStates=1,
                     return_state=True)
print('Initial Energy = {}'.format(energy_gs))

# Create an MPO with Energy subtracted
hamParams = (Omega_f,Delta_f,V_f)
mpo = return_mpo(N,hamParams,D=Dmpo)

# Run DMRG again 
energy_gs,_ = dmrg(mpo,
                   mbd=10,
                   left=False,
                   nStates=1,
                   return_state=True)

# Create an MPO with Energy subtracted
hamParams = (Omega_f,Delta_f,V_f)
mpo = return_mpo(N,hamParams,D=Dmpo,const=real(energy_gs))
mpo = return_mpo(N,hamParams,D=Dmpo)

# Run time evolution
print('Ground State Energy = {}'.format(energy_gs))
E = tddmrg(mpo,mps,
           dt=0.01,
           tf=tf,
           mbd=mbd)
