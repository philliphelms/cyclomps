import unittest
from cyclomps.tools.utils import *

class test_env(unittest.TestCase):
    
    def test_empty_mpo(self):
        mpiprint(0,'\n'+'='*20+'\nTesting Environment List Creation\n'+'='*20)
        # Create an MPS
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_right
        from cyclomps.tools.env_tools import calc_env,env_load_ten
        d = 3
        N = 10
        mbd = 100
        nState = 4
        mpsList = create_mps_list([d]*N,mbd,nState)
        mpsList = make_mps_list_right(mpsList)
        # Load an mpo
        mpoList = []
        mpo = [None]*N
        mpoList.append(mpo)
        # Calculate the environment
        envList = calc_env(mpsList,mpoList)
        # We can assert that the environment is now an identity at each site
        for site in range(N-1,0,-1):
            ident_check = env_load_ten(envList,0,site)
            ident_check = einsum('ijk->ik',ident_check)
            (nx,ny) = ident_check.shape
            I = eye(nx)
            diff = summ(abss(I-ident_check))
            mpiprint(0,'Difference from identity = {}'.format(diff))
            self.assertTrue(diff < 1e-8)
        # We can check the norm to see if it agrees
        mpiprint(0,'Passed\n'+'='*20)

