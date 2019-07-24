import unittest
from cyclomps.tools.utils import *

class test_algs(unittest.TestCase):
    
    def test_dmrg1(self):
        mpiprint(0,'\n'+'='*20+'\nTesting one-site dmrg\n'+\
                 '-Exact Diagonalization\n-not fixed\n-not orthonormalized\n-calc left\n'+'='*20)
        # Use the TASEP MPO
        from cyclomps.algs.dmrg1 import dmrg
        from numpy import complex_
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(10,(0.5,0.5,1.))
        E0 = dmrg(mpo,
                  alg='exact',
                  dtype=complex_,
                  tol=1e-3,
                  max_iter=[4,2],
                  min_iter=[2,0],
                  mbd = [10,20],
                  mps_subdir='test_mps',
                  env_subdir='test_env',
                  nStates=1,
                  fixed_bd=False,
                  orthonormalize=False,
                  left=True,
                  end_gauge=5)
        mpiprint(0,'Passed\n'+'='*20)

    def test_dmrg2(self):
        mpiprint(0,'\n'+'='*20+'\nTesting one-site dmrg\n'+\
                 '-Davidson\n-Fixed BD\n-not orthonormalized\n-calc left\n-Multiple BD\n'+'='*20)
        # Use the TASEP MPO
        from cyclomps.algs.dmrg1 import dmrg
        from numpy import complex_
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(10,(0.5,0.5,1.))
        E1 = dmrg(mpo,
                  alg='davidson',
                  dtype=complex_,
                  tol=1e-5,
                  mbd=[10,20],
                  min_iter=3,
                  nStates=2,
                  fixed_bd=True,
                  left=False)
        mpiprint(0,'Passed\n'+'='*20)
        
    """
    def test_dmrg3(self):
        mpiprint(0,'\n'+'='*20+'\nTesting one-site dmrg\n'+\
                 '-Arnoldi\n-Fixed BD\n-orthonormalized\n-noise added\n-Multiple BD\n'+'='*20)
        # Use the TASEP MPO
        from cyclomps.algs.dmrg1 import dmrg
        from numpy import complex_
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(10,(0.5,0.5,1.))
        E2 = dmrg(mpo,
                  alg='arnoldi',
                  dtype=complex_,
                  tol=[1e-5],
                  max_iter=3,
                  min_iter=0,
                  mbd=5,
                  nStates=2,
                  orthonormalize=True,
                  noise=True)
        mpiprint(0,'Passed\n'+'='*20)
    """

if __name__ == "__main__":
    unittest.main()
