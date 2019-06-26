import unittest
from cyclomps.tools.utils import *

class test_algs(unittest.TestCase):
    
    def test_dmrg1(self):
        mpiprint(0,'\n'+'='*20+'\nTesting one-site dmrg\n'+'='*20)
        # Use the TASEP MPO
        from cyclomps.algs.dmrg1 import dmrg
        from numpy import complex_
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(10,(0.5,0.5,1.))
        E0 = dmrg(mpo,
                  alg='exact',
                  dtype=complex_,
                  tol=1e-3,
                  max_iter=4,
                  min_iter=2,
                  mps_subdir='test_mps',
                  env_subdir='test_env',
                  nStates=3,
                  fixed_bd=False,
                  orthonormalize=False,
                  end_gauge=5)

        E1 = dmrg(mpo,
                  alg='davidson',
                  dtype=complex_,
                  tol=1e-5,
                  mbd=100,
                  min_iter=10,
                  nStates=1,
                  fixed_bd=True,
                  left=True)
        
        E2 = dmrg(mpo,
                  alg='arnoldi',
                  dtype=complex_,
                  tol=[1e-5],
                  max_iter=3,
                  min_iter=0,
                  mbd=50,
                  nStates=4,
                  orthonormalize=True,
                  noise=True)
        
        # We can check the norm to see if it agrees
        mpiprint(0,'Passed\n'+'='*20)
