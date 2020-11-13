from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex_
from cyclomps.mpo.fa import return_mpo
import cProfile
import pstats

# System Size
N = 20

# Hamiltonian Parameters
c = 0.2
s = 1.

# Maximum bond dimension
mbd = [10,100]

# Specify number of processors
n = 28 # Only for correctly naming profile results

# Calculation Settings
alg = 'davidson'
tol = 1e-5
max_iter = 1
min_iter = 1
mps_dir = 'fa_mps'
env_dir = 'fa_env'
nStates = 2
fixed_bd = True
state_avg = True
orthonormalize = False
end_gauge = 0
left = False

# Set up mpo
mpo = return_mpo(N,(c,s))

# Run DMRG
E0 = dmrg(mpo,
          alg=alg,
          dtype=complex_,
          mbd = mbd,
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
