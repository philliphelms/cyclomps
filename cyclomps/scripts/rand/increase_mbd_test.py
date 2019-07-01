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
              mbd = [10,100,500],#2,10,50,100,500,1000,5000,10000,25000,50000,75000,100000],
              tol=1e-5,
              max_iter=[1,1,1000000],
              min_iter=[0,0,999999],
              mps_subdir='mbd_test_mps',
              env_subdir='mbd_test_env',
              nStates=1,
              fixed_bd=True,
              state_avg=False,
              orthonormalize=False,
              end_gauge=0,
              left=False)

if __name__ == "__main__":
    test_dmrg1()
