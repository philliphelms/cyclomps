import unittest
from cyclomps.tools.utils import *
from cyclomps.tools.mps_tools import *

class test_canonicalization(unittest.TestCase):
    
    def test_create_mps_list(self):
        mpiprint(0,'\n'+'='*50+'\nTesting MPS List Creation\n'+'-'*50)
        mpsL = create_mps_list([3]*10,100,4)
        self.assertEqual(len(mpsL),4)
        self.assertEqual(len(mpsL[0]),10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_canonicalize_mps1(self):
        mpiprint(0,'\n'+'='*50+'\nTesting Canonicalization effect on Config Probs\n'+'='*20)
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
        self.assertTrue(summ(abss(val1_init-val1_fin)) < 1e-6)
        self.assertTrue(summ(abss(val0_init-val0_fin)) < 1e-6)
        mpiprint(0,'Passed\n'+'='*50)

    def test_canonicalize_mps2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting Correct Canonical Form\n'+'-'*50)
        N = 10
        d = 3
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        # Make the MPS Right canonical
        mpsL = make_mps_list_right(mpsL)
        # Ensure the canonicalization is correct
        for state in range(nState):
            for site in range(N-1,0,-1):
                # Load Tensor of interest
                ten = mps_load_ten(mpsL,state,site)
                # Compute the isometry
                ident_check = einsum('ijk,ljk',ten,conj(ten))
                # Compare with the identity
                (nx,ny) = ident_check.shape
                I = eye(nx)
                diff = summ(abss(I-ident_check))
                self.assertTrue(diff < 1e-8)
        mpiprint(0,'Passed\n'+'='*50)

    def test_canonicalize_mps3(self):
        mpiprint(0,'\n'+'='*50+'\nTesting Canonicalization effect on Norm\n'+'-'*50)
        N = 4
        d = 2
        mbd = 10
        nState = 5
        mpsL = create_mps_list([d]*N,mbd,nState)
        # Calculate initial norm
        norm = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm[state] = calc_mps_norm(mpsL,state=state)
        # Make the MPS Right canonical
        mpsL = make_mps_list_right(mpsL)
        # Calculate Final norm
        norm2 = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm2[state] = calc_mps_norm(mpsL,state=state)
        # Print
        for state in range(nState):
            mpiprint(0,'Norm of State {}: {},{}'.format(state,norm[state],norm2[state]))
            self.assertTrue(abs(norm[state]-norm2[state]) < 1e-8)
        mpiprint(0,'Passed\n'+'='*50)

    def test_canonicalize_mps4(self):
        mpiprint(0,'\n'+'='*50+'\nTesting Canonicalization effect on Energy\n'+'-'*50)
        N = 4
        d = 2
        mbd = 10
        nState = 5
        mpsL = create_mps_list([d]*N,mbd,nState)
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create function to calculate energy and norm
        def calc_energy(mps,mpo,load_mps=True):
            if load_mps:
                mps = load_mps_list(mps)
            E = zeros(nState,dtype=mps[0][0].dtype)
            norm = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                norm[state] = einsum('pqrs,pqrs->',psi,psi_conj)
            return E,norm
        # Calculate initial norm
        E,norm = calc_energy(mpsL,mpo,load_mps=True)
        # Make the MPS Right canonical
        mpsL = make_mps_list_left(mpsL)
        # Calculate Final norm
        E2,norm2 = calc_energy(mpsL,mpo,load_mps=True)
        # Print
        for state in range(nState):
            mpiprint(0,'Norm of State {}: {},{},{}'.format(state,norm[state],norm2[state],abs(norm[state]-norm2[state])))
            self.assertTrue(abs(norm[state]-norm2[state]) < 1e-8)
            mpiprint(0,'Energy of State {}: {},{},{}'.format(state,E[state],E2[state],abs(E[state]-E2[state])))
            self.assertTrue(abs(E[state]-E2[state]) < 1e-8)
        mpiprint(0,'Passed\n'+'='*50)

    def test_move_gauge_right(self):
        mpiprint(0,'\n'+'='*50+'\nTesting moving gauge right effect on Norm\n'+'-'*50)
        N = 4
        d = 2
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        # Calculate initial norm
        norm = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm[state] = calc_mps_norm(mpsL,state=state)
        # Move guage one site left
        mpsL = move_gauge_right(mpsL,0,0,return_ent=False,return_wgt=False)
        # Calculate Final norm
        norm2 = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm2[state] = calc_mps_norm(mpsL,state=state)
        # Print
        for state in range(nState):
            mpiprint(0,'Norm of State {}: {},{}'.format(state,norm[state],norm2[state]))
            self.assertTrue(abs(norm[state]-norm2[state]) < 1e-8)
        mpiprint(0,'Passed\n'+'='*50)

    def test_move_gauge_right2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting moving gauge right effect on Energy\n'+'-'*50)
        N = 4
        d = 2
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Calculate initial Energy (very slow way)
        mps = load_mps_list(mpsL)
        E = zeros(nState,dtype=mps[0][0].dtype)
        for state in range(nState):
            psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
            op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
            psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
            E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
        # Move guage one site left
        for state in range(nState):
            mpsL = move_gauge_right(mpsL,state,0,return_ent=False,return_wgt=False)
        # Calculate Final Energy
        mps = load_mps_list(mpsL)
        E2 = zeros(nState,dtype=mps[0][0].dtype)
        for state in range(nState):
            psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
            op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
            psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
            E2[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
        # Print
        for state in range(nState):
            mpiprint(0,'State {} Energies: {},{},{}'.format(state,E[state],E2[state],summ(abss(E[state]-E2[state]))))
            self.assertTrue(abs(E[state]-E2[state]) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_move_gauge_left(self):
        mpiprint(0,'\n'+'='*50+'\nTesting moving gauge left effect on Norm\n'+'-'*50)
        N = 4
        d = 2
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        # Calculate initial norm
        norm = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm[state] = calc_mps_norm(mpsL,state=state)
        # Move guage one site left
        for state in range(nState):
            mpsL = move_gauge_left(mpsL,state,3,return_ent=False,return_wgt=False)
        # Calculate Final norm
        norm2 = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm2[state] = calc_mps_norm(mpsL,state=state)
        # Print
        for state in range(nState):
            mpiprint(0,'Norm of State {}: {},{}'.format(state,norm[state],norm2[state]))
            self.assertTrue(abs(norm[state]-norm2[state]) < 1e-8)
        mpiprint(0,'Passed\n'+'='*50)

    def test_move_gauge_left_(self):
        mpiprint(0,'\n'+'='*50+'\nTesting moving gauge left effect on Norm\n'+'-'*50)
        from cyclomps.mpo.tasep import return_mpo
        N = 4
        d = 2
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Calculate initial norm
        mps = load_mps_list(mpsL)
        norm = zeros(nState,dtype=mpsL[0][0]['dtype'])
        E = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm[state] = calc_mps_norm(mpsL,state=state)
            psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
            op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
            psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
            E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
        # Move guage one site left
        for state in range(nState):
            mpsL = move_gauge_left(mpsL,state,3,return_ent=False,return_wgt=False)
        # Calculate Final norm
        mps2 = load_mps_list(mpsL)
        norm2 = zeros(nState,dtype=mpsL[0][0]['dtype'])
        E_ = zeros(nState,dtype=mpsL[0][0]['dtype'])
        E2 = zeros(nState,dtype=mpsL[0][0]['dtype'])
        for state in range(nState):
            norm2[state] = calc_mps_norm(mpsL,state=state)
            psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
            op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
            psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
            E_[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
            psi = einsum('apb,bqc,crd,dse->pqrs',mps2[state][0],mps2[state][1],mps2[state][2],mps2[state][3])
            op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
            psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps2[state][0]),conj(mps2[state][1]),conj(mps2[state][2]),conj(mps2[state][3]))
            E2[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
        # Print
        for state in range(nState):
            mpiprint(0,'Norm of State {}: {},{}'.format(state,norm[state],norm2[state]))
            mpiprint(0,'Energies: {}, {}, {}'.format(E[state],E_[state],E2[state]))
            self.assertTrue(abs(norm[state]-norm2[state]) < 1e-8)
            self.assertTrue(abs(E[state]-E2[state]) < 1e-8)
        mpiprint(0,'Passed\n'+'='*50)

    def test_move_gauge_left2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting moving gauge left effect on Energy\n'+'-'*50)
        from cyclomps.mpo.tasep import return_mpo
        import copy
        N = 4
        d = 2
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Calculate initial Energy (very slow way)
        mps = load_mps_list(mpsL)
        E = zeros(nState,dtype=mps[0][0].dtype)
        for state in range(nState):
            psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
            op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
            psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
            E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
            #E[state] /= calc_mps_norm(mpsL,state)
        # Move guage one site left
        for state in range(nState):
            site = 3
            ten1 = mps_load_ten(mpsL,state,site-1)
            ten2 = mps_load_ten(mpsL,state,site)
            ten1,ten2,_,_,_ = move_gauge_left_tens(copy.deepcopy(ten1),copy.deepcopy(ten2))
            mps_save_ten(ten1,mpsL,state,site-1)
            mps_save_ten(ten2,mpsL,state,site)
        #for state in range(nState):
        #    mpsL = move_gauge_left(mpsL,state,3,return_ent=False,return_wgt=False)
        # Calculate Final Energy
        mps = load_mps_list(mpsL)
        E2 = zeros(nState,dtype=mps[0][0].dtype)
        for state in range(nState):
            psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
            op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
            psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
            E2[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
            #E2[state] /= calc_mps_norm(mpsL,state)
        # Print
        for state in range(nState):
            self.assertTrue(abs(E[state]-E2[state]) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_save_mps(self):
        mpiprint(0,'\n'+'='*50+'\nTesting saving mps\n'+'-'*50)
        # Create an MPS
        N = 4
        d = 2
        mbd = 10
        nState = 2
        mpsL = create_mps_list([d]*N,mbd,nState)
        # Change one Element of the mps
        ten = mps_load_ten(mpsL,0,2)
        dims = ten.shape
        ten_replace = rand(dims,dtype=ten.dtype)
        # Save result
        mps_save_ten(ten_replace,mpsL,0,2)
        # Reload Result
        ten_replace_reloaded = mps_load_ten(mpsL,0,2)
        # Calculate difference
        diff = summ(abss(ten_replace-ten_replace_reloaded))
        self.assertTrue(diff < 1e-8)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
