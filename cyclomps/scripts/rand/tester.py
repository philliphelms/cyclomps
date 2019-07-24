from cyclomps.tools.utils import *

mpiprint(0,'\n'+'='*20+'\nTesting one-site dmrg\n'+'='*20)
# Use the TASEP MPO
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex_
from cyclomps.mpo.asep import return_mpo
mpo = return_mpo(10,(0.5,0.5,0.1,0.9,0.5,0.5,-0.5))
E0,mps = dmrg(mpo,
          alg='davidson',
          dtype=complex_,
          tol=1e-3,
          mbd = 10,
          max_iter=20,
          min_iter=10,
          mps_subdir='test_mps',
          env_subdir='test_env',
          nStates=3,
          fixed_bd=True,
          orthonormalize=False,
          return_state = True,
          state_avg = True)

E0 = dmrg(mpo,
          mps = mps,
          alg='davidson',
          dtype=complex_,
          tol=1e-3,
          mbd = [10,20,30],
          max_iter=2,
          min_iter=0,
          mps_subdir='test_mps',
          env_subdir='test_env',
          nStates=3,
          fixed_bd=True,
          orthonormalize=False,
          return_state = False,
          state_avg = True)
