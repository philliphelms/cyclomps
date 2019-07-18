import unittest
from cyclomps.tools.utils import *

class test_diag(unittest.TestCase):
    
    def test_retrieve(self):
        mpiprint(0,'\n'+'='*20+'\nTesting Tensor Retrieve Function\n'+'='*20)
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
        # Retrieve Relevant tensors for the local optimization
        (mps0,) = retrieve_tensors(0,mpsList=mpsList)
        (mps1,) = retrieve_tensors(1,mpsList=mpsList)
        (envL,envR) = retrieve_tensors(0,envList=envList)
        (env2L,env2R) = retrieve_tensors(0,envList=envList,twoSite=True)
        (mpo0,) = retrieve_tensors(0,mpoList=mpoList)
        (mpo1,) = retrieve_tensors(1,mpoList=mpoList)
        # Check to make sure all dimensions align
        (n1,n2,n3) = mps0[0].shape
        (n4,n5,n6) = mps1[0].shape
        (n7,n8,n9) = envL[0].shape
        (n10,n11,n12) = envR[0].shape
        (n13,n14,n15) = env2L[0].shape
        (n16,n17,n18) = env2R[0].shape
        self.assertEqual(n1,n7)
        self.assertEqual(n1,n13)
        self.assertEqual(n3,n4)
        self.assertEqual(n3,n12)
        self.assertEqual(n3,n10)
        self.assertEqual(n6,n16)
        self.assertEqual(n6,n18)
        # We can check the norm to see if it agrees
        mpiprint(0,'Passed\n'+'='*20)

    def test_one_site(self):
        mpiprint(0,'\n'+'='*20+'\nTesting One Site Optimization\n'+'='*20)
        # Create an MPS
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_right
        from cyclomps.tools.env_tools import calc_env,env_load_ten
        from cyclomps.tools.diag_tools import eig1
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
        # Retrieve Relevant tensors for the local optimization
        (mps,) = retrieve_tensors(0,mpsList=mpsList)
        (envl,envr) = retrieve_tensors(0,envList=envList)
        (mpo,) = retrieve_tensors(0,mpoList=mpoList)
        # Send to diagonalization routine
        E,mps0,ovlp = eig1(mps,mpo,envl,envr,alg='exact')
        mpiprint(0,'Exact Worked')
        E,mps0,ovlp = eig1(mps,mpo,envl,envr,alg='arnoldi')
        mpiprint(0,'Arnoldi Worked')
        E,mps0,ovlp = eig1(mps,mpo,envl,envr,alg='davidson')
        mpiprint(0,'Davidson Worked')
        mpiprint(0,'Passed\n'+'='*20)

    #def test_two_site(self):
    #    self.assertTrue(False)
if __name__ == "__main__":
    unittest.main()
