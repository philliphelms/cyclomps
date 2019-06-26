import unittest
from cyclomps.tools.utils import *

def test_dmrg1():
    mpiprint(0,'\n'+'='*20+'\nTesting one-site dmrg\n'+'='*20)
    # Use the TASEP MPO
    from cyclomps.algs.dmrg1 import dmrg
    from numpy import complex_
    from cyclomps.mpo.tasep import return_mpo
    mpo = return_mpo(10,(0.5,0.5,1.))
    E0 = dmrg(mpo,
              alg='davidson',
              dtype=complex_,
              mbd = [2,10],
              tol=1e-3,
              max_iter=4,
              min_iter=0,
              mps_subdir='test_mps',
              env_subdir='test_env',
              nStates=2,
              fixed_bd=True,
              state_avg=True,
              orthonormalize=False,
              end_gauge=0,
              left=True)

    """
    E1 = dmrg(mpo,
              alg='exact',
              dtype=complex_,
              tol=1e-5,
              mbd=10,
              min_iter=3,
              max_iter=5,
              nStates=1,
              fixed_bd=True,
              state_avg=False,
              left=False)
    
    E2 = dmrg(mpo,
              alg='arnoldi',
              dtype=complex_,
              tol=[1e-5],
              max_iter=3,
              min_iter=0,
              mbd=50,
              nStates=4,
              fixed_bd=True,
              state_avg=True,
              orthonormalize=True,
              noise=True)
    
    # We can check the norm to see if it agrees
    mpiprint(0,'Passed\n'+'='*20)
    """

if __name__ == "__main__":
    test_dmrg1()
