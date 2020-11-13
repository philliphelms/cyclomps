from numpy import float_
from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.algs.ed import ed
from cyclomps.mpo.rydberg import return_mpo
#from sys import argv
#from numpy import logspace,linspace 

# System Size
N = 10
mbd = 50

# Hamiltonian Parameters
Omega = 0.1
Delta = 0.001
V     = 1.

# Calculate Settings
alg = 'davidson'
tol = 1e-5
max_iter = 5
min_iter = 1 
mps_dir = 'rydberg_mps'
env_dir = 'rydberg_env'
nStates = 1
fixed_bd = False
state_avg = False
orthonormalize = False
end_gauge = 0 
left = False


#s = sVec[0]
hamParams = (Omega,Delta,V)
mpo = return_mpo(N,hamParams)

if False:
    # Run exact diagonalization
    u,v = ed(mpo)
    print(u)

# Run diagonalization
E0 = dmrg(mpo,
          alg=alg,
          dtype=float_,
          mbd =mbd,
          tol=tol,
          max_iter=max_iter,
          min_iter=min_iter,
          mps_subdir=mps_dir,
          env_subdir=env_dir, 
          nStates=nStates,
          fixed_bd=fixed_bd,
          state_avg=state_avg,
          orthonormalize=orthonormalize,
          end_gauge=end_gauge,
          left=left)
