import unittest
from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex_
from cyclomps.mpo.tasep import return_mpo
import cProfile
import pstats

# System Size
N = 100

# Hamiltonian Parameters
alpha = 0.5
beta = 0.5
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
mps_dir = tasep_mps
env_dir = tasep_env
nStates = 1
fixed_bd = True
state_avg = False
orthonormalize = False
end_gauge = 0
left = False

# Set up mpo
mpo = return_mpo(10,(alpha,beta,s))

# Run DMRG
cProfile.run("\
        E0 = dmrg(mpo,\
                  alg="+alg",\
                  dtype=complex_,mbd = "+str(mbd)+",\
                  tol="+str(tol)",\
                  max_iter="+str(max_iter)",\
                  min_iter="+str(min_iter)",\
                  mps_subdir="+mps_dir ",\
                  env_subdir="+env_dir ",\
                  nStates="+str(nStates)",\
                  fixed_bd="+str(fixed_bd)",\
                  state_avg="+str(state_avg)",\
                  orthonormalize="+str(orthonormalize)",\
                  end_gauge="+str(end_gauge)",\
                  left="+str(left)")",
                  'n'+str(n)+'_dmrg_stats_'+str(mbdVec))
if RANK == 0:
    p = pstats.Stats('n'+str(n)+'_dmrg_stats_'+str(mbdVec))
