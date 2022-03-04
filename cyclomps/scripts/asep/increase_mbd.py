from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex_
from cyclomps.mpo.asep import return_mpo
import cProfile
import pstats

# System Size
N = 10

# Hamiltonian Parameters
alpha = 0.5
gamma = 0.5
p = 0.1
q = 1.-p
beta = 0.5
delta = 0.5
s = 1.

# Maximum bond dimension
mbd = [10,100]

# Specify number of processors
n = 28

# Calculation Settings
alg = 'davidson'
tol = 1e-5
max_iter = 1
min_iter = 1
mps_dir = 'asep_mps'
env_dir = 'asep_env'
nStates = 1
fixed_bd = True
state_avg = False
orthonormalize = False
end_gauge = 0
left = False

# Set up mpo
mpo = return_mpo(10,(alpha,gamma,p,q,beta,delta,s))

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
