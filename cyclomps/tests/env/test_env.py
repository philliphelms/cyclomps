import unittest
from cyclomps.tools.utils import *

class test_env(unittest.TestCase):
    
    def test_empty_mpo(self):
        mpiprint(0,'\n'+'='*50+'\nTesting Initial Environment Using Identity MPO\n'+'-'*50)
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
            self.assertTrue(diff < 1e-8)
        # We can check the norm to see if it agrees
        mpiprint(0,'Passed\n'+'='*50)

    def test_tasep_mpo(self):
        mpiprint(0,'\n'+'='*50+'\nTesting Initial Environment Using TASEP MPO\n'+'-'*50)
        # Create an MPS
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_right,load_mps_list,calc_mps_norm
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mpsList = create_mps_list([d]*N,mbd,nState)
        # Load the MPO
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy 
        def calc_energy(mps,mpo,load_mps=True):
            if load_mps:
                mps = load_mps_list(mps)
            E = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                norm = einsum('pqrs,pqrs->',psi,psi_conj)
                E[state] /= norm
            return E
        # Compute energy
        E1 = calc_energy(mpsList,mpo)
        mpiprint(0,'Initial Energy =\n{}'.format(E1))
        # Right Canonicalize
        mpsList = make_mps_list_right(mpsList)
        # Compute energy again
        E2 = calc_energy(mpsList,mpo)
        mpiprint(0,'Energy after canonicalization =\n{}'.format(E2))
        # Calculate the environment
        from cyclomps.tools.env_tools import calc_env,env_load_ten,load_env_list
        envList = calc_env(mpsList,mpo)
        # Calculate the energy using the environment
        env = load_env_list(envList)
        mps = load_mps_list(mpsList)
        E3 = zeros(nState,dtype=mps[0][0].dtype)
        for state in range(nState):
            E3[state] = einsum('Ala,apb,lPpr,APB,Brb->',env[0][0],mps[0][0],mpo[0][0],conj(mps[0][0]),env[0][1])
            E3[state] /= calc_mps_norm(mpsList,state=state)
        mpiprint(0,'Energy from environment =\n{}'.format(E3))
        # Check Results
        self.assertTrue(summ(abss(E1[0]-E2[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[0]-E3[0])) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_tasep_mpo2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting Initial & Moved Environments Using TASEP MPO\n'+'-'*50)
        # Create an MPS
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_right,load_mps_list,calc_mps_norm,move_gauge
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mpsList = create_mps_list([d]*N,mbd,nState)
        # Load the MPO
        from cyclomps.mpo.tasep import return_mpo
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy 
        def calc_energy(mps,mpo,load_mps=True):
            if load_mps:
                mps = load_mps_list(mps)
            E = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                norm = einsum('pqrs,pqrs->',psi,psi_conj)
                E[state] /= norm
            return E
        # Compute energy
        E1 = calc_energy(mpsList,mpo)
        # Right Canonicalize
        mpsList = make_mps_list_right(mpsList)
        E2 = calc_energy(mpsList,mpo)
        # Calculate the environment
        from cyclomps.tools.env_tools import calc_env,env_load_ten,load_env_list
        envList = calc_env(mpsList,mpo,gSite=0)
        # Calc energy from initial environment
        env = load_env_list(envList)
        mps = load_mps_list(mpsList)
        E3 = zeros(nState,dtype=mps[0][0].dtype)
        for state in range(nState):
            E3[state] = einsum('Ala,apb,lPpr,APB,Brb->',env[0][0],mps[0][0],mpo[0][0],conj(mps[0][0]),env[0][1])
            E3[state] /= calc_mps_norm(mpsList,state=state)
        # Calculate the energy using the environment
        mpsList = move_gauge(mpsList,0,2)
        envList = calc_env(mpsList,mpo,gSite=2)
        env = load_env_list(envList)
        mps = load_mps_list(mpsList)
        E4 = zeros(nState,dtype=mps[0][0].dtype)
        for state in range(nState):
            E4[state] = einsum('Ala,apb,lPpr,APB,Brb->',env[0][2],mps[0][2],mpo[0][2],conj(mps[0][2]),env[0][3])
            E4[state] /= calc_mps_norm(mpsList,state=state)
        # Check Results
        self.assertTrue(summ(abss(E1[0]-E2[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[0]-E3[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[0]-E4[0])) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
