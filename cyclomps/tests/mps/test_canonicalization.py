import unittest
from cyclomps.tools.utils import *
from cyclomps.tools.mps_tools import *

class test_canonicalization(unittest.TestCase):
    
    def test_create_mps_list(self):
        mpiprint(0,'\n'+'='*20+'\nTesting MPS List Creation\n'+'='*20)
        mpsL = create_mps_list([3]*10,100,4)
        self.assertEqual(len(mpsL),4)
        self.assertEqual(len(mpsL[0]),10)
        mpiprint(0,'Passed\n'+'='*20)

    def test_canonicalize_mps(self):
        mpiprint(0,'\n'+'='*20+'\nTesting Right Canonicalization Procedure\n'+'='*20)
        N = 10
        d = 3
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        # Calculate the probabilities associated with a few states
        val0_init = contract_config(mpsL,[0]*N)
        val1_init = contract_config(mpsL,[1]*N)
        # Make the MPS Right canonical
        mpsL = make_mps_list_right(mpsL)
        # Recalculate the probabilities and compare to make sure they are the same
        val0_fin = contract_config(mpsL,[0]*N)
        val1_fin = contract_config(mpsL,[1]*N)
        mpiprint(0,'Difference between empty config probabilities = |{}-{}| = {}'.format(val0_init,val0_fin,abss(val0_init-val0_fin)))
        mpiprint(0,'Difference between full config probabilities = |{}-{}| = {}'.format(val1_init,val1_fin,abss(val1_init-val1_fin)))
        self.assertTrue(abss(val1_init-val1_fin) < 1e-8)
        self.assertTrue(abss(val0_init-val0_fin) < 1e-8)
        # Ensure the canonicalization is correct
        for state in range(nState):
            for site in range(N-1,0,-1):
                mpiprint(0,'Check Canonicalization state {} site {}'.format(state,site))
                # Load Tensor of interest
                ten = mps_load_ten(mpsL,state,site)
                # Compute the isometry
                ident_check = einsum('ijk,ljk',ten,conj(ten))
                # Compare with the identity
                (nx,ny) = ident_check.shape
                I = eye(nx)
                diff = summ(abss(I-ident_check))
                mpiprint(0,'Difference from identity = {}'.format(diff))
                self.assertTrue(diff < 1e-8)
        mpiprint(0,'Passed\n'+'='*20)
