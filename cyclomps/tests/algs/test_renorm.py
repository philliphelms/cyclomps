import unittest
from cyclomps.tools.utils import *
import copy

class test_algs(unittest.TestCase):
        
    def test_renorm_right(self):
        mpiprint(0,'\n'+'='*50+'\nTesting right renormalization\n'+'-'*50)
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_right,load_mps_list,calc_mps_norm,move_gauge
        from cyclomps.mpo.tasep import return_mpo
        from cyclomps.algs.dmrg1 import renormalize_right
        # Create an MPS
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mps = create_mps_list([d]*N,mbd,nState)
        mps = make_mps_list_right(mps)
        # Load the MPO
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy
        def calc_energy(mpsL,mpo):
            mps = load_mps_list(mpsL)
            E = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                E[state] /= calc_mps_norm(mpsL,state=state)
            return E
        # Calc initial Energy
        E1 = calc_energy(mps,mpo)
        # Move gauge using renormalization
        site = 0
        (mps0,) = retrieve_tensors(site,mpsList=mps)
        (mps1,) = retrieve_tensors(site+1,mpsList=mps)
        mps0,mps1,EE,EEs,wgt = renormalize_right(mps0,mps1,state_avg=False)
        save_tensors(site,mpsList=mps,mps=mps0)
        save_tensors(site+1,mpsList=mps,mps=mps1)
        # Calc Energy Again
        E2 = calc_energy(mps,mpo)
        # Check Energies
        self.assertTrue(summ(abss(E1[0]-E2[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[1]-E2[1])) < 1e-10)
        self.assertTrue(summ(abss(E1[2]-E2[2])) < 1e-10)
        self.assertTrue(summ(abss(E1[3]-E2[3])) < 1e-10)
        # Check for correct canonical form
        for state in range(nState):
            eye_check = einsum('apb,apc->bc',mps0[state],conj(mps0[state]))
            eye_actual= eye(eye_check.shape[0])
            self.assertTrue(summ(abss(eye_check-eye_actual)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)
        
    def test_renorm_right2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting right state averaged renormalization\n'+'-'*50)
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_right,load_mps_list,calc_mps_norm,move_gauge
        from cyclomps.mpo.tasep import return_mpo
        from cyclomps.algs.dmrg1 import renormalize_right
        # Create an MPS
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mps = create_mps_list([d]*N,mbd,nState)
        mps = make_mps_list_right(mps)
        # Load the MPO
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy
        def calc_energy(mpsL,mpo):
            mps = load_mps_list(mpsL)
            E = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                E[state] /= calc_mps_norm(mpsL,state=state)
            return E
        # Calc initial Energy
        E1 = calc_energy(mps,mpo)
        # Move gauge using renormalization
        site = 0
        (mps0,) = retrieve_tensors(site,mpsList=mps)
        (mps1,) = retrieve_tensors(site+1,mpsList=mps)
        mps0,mps1,EE,EEs,wgt = renormalize_right(mps0,mps1,state_avg=True)
        save_tensors(site,mpsList=mps,mps=mps0)
        save_tensors(site+1,mpsList=mps,mps=mps1)
        # Calc Energy Again
        E2 = calc_energy(mps,mpo)
        # Check Energies
        self.assertTrue(summ(abss(E1[0]-E2[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[1]-E2[1])) < 1e-10)
        self.assertTrue(summ(abss(E1[2]-E2[2])) < 1e-10)
        self.assertTrue(summ(abss(E1[3]-E2[3])) < 1e-10)
        # Check for correct canonical form
        for state in range(nState):
            eye_check = einsum('apb,apc->bc',mps0[state],conj(mps0[state]))
            eye_actual= eye(eye_check.shape[0])
            self.assertTrue(summ(abss(eye_check-eye_actual)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_renorm_left_norm(self):
        mpiprint(0,'\n'+'='*50+'\nTesting left renormalization effect on norm\n'+'-'*50)
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_left,load_mps_list,calc_mps_norm,move_gauge
        from cyclomps.mpo.tasep import return_mpo
        from cyclomps.algs.dmrg1 import renormalize_left
        # Create an MPS
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mps = create_mps_list([d]*N,mbd,nState)
        mps = make_mps_list_left(mps)
        # Load the MPO
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy
        def calc_energy(mpsL,mpo):
            mps = load_mps_list(mpsL)
            E = zeros(nState,dtype=mps[0][0].dtype)
            norm = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                norm[state] = calc_mps_norm(mpsL,state=state)
            return E,norm
        # Calc initial Energy
        E1,norm1 = calc_energy(mps,mpo)
        # Move gauge using renormalization
        site = N-1
        (mps0,) = retrieve_tensors(site-1,mpsList=mps)
        (mps1,) = retrieve_tensors(site,mpsList=mps)
        mps0,mps1,EE,EEs,wgt = renormalize_left(mps0,mps1,state_avg=False)
        save_tensors(site-1,mpsList=mps,mps=mps0)
        save_tensors(site,mpsList=mps,mps=mps1)
        # Calc Energy Again
        E2,norm2 = calc_energy(mps,mpo)
        # Check Energies
        self.assertTrue(summ(abss(norm1[0]-norm2[0])) < 1e-10)
        self.assertTrue(summ(abss(norm1[1]-norm2[1])) < 1e-10)
        self.assertTrue(summ(abss(norm1[2]-norm2[2])) < 1e-10)
        self.assertTrue(summ(abss(norm1[3]-norm2[3])) < 1e-10)
        # Check for correct canonical form
        for state in range(nState):
            eye_check = einsum('apb,cpb->ac',mps1[state],conj(mps1[state]))
            eye_actual= eye(eye_check.shape[0])
            self.assertTrue(summ(abss(eye_check-eye_actual)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_renorm_left_energy(self):
        mpiprint(0,'\n'+'='*50+'\nTesting left renormalization effect on Energy\n'+'-'*50)
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_left,load_mps_list,calc_mps_norm,move_gauge
        from cyclomps.mpo.tasep import return_mpo
        from cyclomps.algs.dmrg1 import renormalize_left
        # Create an MPS
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mps = create_mps_list([d]*N,mbd,nState)
        mps = make_mps_list_left(mps)
        # Load the MPO
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy
        def calc_energy(mpsL,mpo):
            mps = load_mps_list(mpsL)
            E = zeros(nState,dtype=mps[0][0].dtype)
            norm = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                norm[state] = calc_mps_norm(mpsL,state=state)
            return E,norm
        # Calc initial Energy
        E1,norm1 = calc_energy(mps,mpo)
        # Move gauge using renormalization
        site = N-1
        (mps0,) = retrieve_tensors(site-1,mpsList=mps)
        (mps1,) = retrieve_tensors(site,mpsList=mps)
        mps0,mps1,EE,EEs,wgt = renormalize_left(mps0,mps1,state_avg=False)
        save_tensors(site-1,mpsList=mps,mps=mps0)
        save_tensors(site,mpsList=mps,mps=mps1)
        # Calc Energy Again
        E2,norm2 = calc_energy(mps,mpo)
        # Check Energies
        self.assertTrue(summ(abss(E1[0]-E2[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[1]-E2[1])) < 1e-10)
        self.assertTrue(summ(abss(E1[2]-E2[2])) < 1e-10)
        self.assertTrue(summ(abss(E1[3]-E2[3])) < 1e-10)
        # Check for correct canonical form
        for state in range(nState):
            eye_check = einsum('apb,cpb->ac',mps1[state],conj(mps1[state]))
            eye_actual= eye(eye_check.shape[0])
            self.assertTrue(summ(abss(eye_check-eye_actual)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_renorm_avg_left_norm(self):
        mpiprint(0,'\n'+'='*50+'\nTesting state-avg left renormalization effect on norm\n'+'-'*50)
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_left,load_mps_list,calc_mps_norm,move_gauge
        from cyclomps.mpo.tasep import return_mpo
        from cyclomps.algs.dmrg1 import renormalize_left
        # Create an MPS
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mps = create_mps_list([d]*N,mbd,nState)
        mps = make_mps_list_left(mps)
        # Load the MPO
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy
        def calc_energy(mpsL,mpo):
            mps = load_mps_list(mpsL)
            E = zeros(nState,dtype=mps[0][0].dtype)
            norm = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                norm[state] = calc_mps_norm(mpsL,state=state)
            return E,norm
        # Calc initial Energy
        E1,norm1 = calc_energy(mps,mpo)
        # Move gauge using renormalization
        site = N-1
        (mps0,) = retrieve_tensors(site-1,mpsList=mps)
        (mps1,) = retrieve_tensors(site,mpsList=mps)
        mps0,mps1,EE,EEs,wgt = renormalize_left(mps0,mps1,state_avg=True)
        save_tensors(site-1,mpsList=mps,mps=mps0)
        save_tensors(site,mpsList=mps,mps=mps1)
        # Calc Energy Again
        E2,norm2 = calc_energy(mps,mpo)
        # Check Energies
        self.assertTrue(summ(abss(norm1[0]-norm2[0])) < 1e-10)
        self.assertTrue(summ(abss(norm1[1]-norm2[1])) < 1e-10)
        self.assertTrue(summ(abss(norm1[2]-norm2[2])) < 1e-10)
        self.assertTrue(summ(abss(norm1[3]-norm2[3])) < 1e-10)
        # Check for correct canonical form
        for state in range(nState):
            eye_check = einsum('apb,cpb->ac',mps1[state],conj(mps1[state]))
            eye_actual= eye(eye_check.shape[0])
            self.assertTrue(summ(abss(eye_check-eye_actual)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_renorm_avg_left_energy(self):
        mpiprint(0,'\n'+'='*50+'\nTesting state avg left renormalization effect on Energy\n'+'-'*50)
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_left,load_mps_list,calc_mps_norm,move_gauge
        from cyclomps.mpo.tasep import return_mpo
        from cyclomps.algs.dmrg1 import renormalize_left
        # Create an MPS
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mps = create_mps_list([d]*N,mbd,nState)
        mps = make_mps_list_left(mps)
        # Load the MPO
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy
        def calc_energy(mpsL,mpo):
            mps = load_mps_list(mpsL)
            E = zeros(nState,dtype=mps[0][0].dtype)
            norm = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
                norm[state] = calc_mps_norm(mpsL,state=state)
            return E,norm
        # Calc initial Energy
        E1,norm1 = calc_energy(mps,mpo)
        # Move gauge using renormalization
        site = N-1
        (mps0,) = retrieve_tensors(site-1,mpsList=mps)
        (mps1,) = retrieve_tensors(site,mpsList=mps)
        mps0,mps1,EE,EEs,wgt = renormalize_left(mps0,mps1,state_avg=True)
        save_tensors(site-1,mpsList=mps,mps=mps0)
        save_tensors(site,mpsList=mps,mps=mps1)
        # Calc Energy Again
        E2,norm2 = calc_energy(mps,mpo)
        # Check Energies
        self.assertTrue(summ(abss(E1[0]-E2[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[1]-E2[1])) < 1e-10)
        self.assertTrue(summ(abss(E1[2]-E2[2])) < 1e-10)
        self.assertTrue(summ(abss(E1[3]-E2[3])) < 1e-10)
        # Check for correct canonical form
        for state in range(nState):
            eye_check = einsum('apb,cpb->ac',mps1[state],conj(mps1[state]))
            eye_actual= eye(eye_check.shape[0])
            self.assertTrue(summ(abss(eye_check-eye_actual)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_renorm_avg_left_energy2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting state avg left renormalization effect on Energy\n'+'-'*50)
        from cyclomps.tools.mps_tools import create_mps_list,make_mps_list_left,load_mps_list,calc_mps_norm,move_gauge
        from cyclomps.mpo.tasep import return_mpo
        from cyclomps.algs.dmrg1 import renormalize_left
        # Create an MPS
        d = 2
        N = 4
        mbd = 10
        nState = 4
        mpsL = create_mps_list([d]*N,mbd,nState)
        mpsL = make_mps_list_left(mpsL)
        # Load the MPO
        mpo = return_mpo(N,(0.5,0.5,1.))
        # Create a function to compute energy
        def calc_energy(mpsL,mpo,load_mps=True):
            if load_mps: 
                mps = load_mps_list(mpsL)
            else:
                mps = mpsL
            E = zeros(nState,dtype=mps[0][0].dtype)
            for state in range(nState):
                psi = einsum('apb,bqc,crd,dse->pqrs',mps[state][0],mps[state][1],mps[state][2],mps[state][3])
                op = einsum('iPpj,jQqk,kRrl,lSsm->PpQqRrSs',mpo[0][0],mpo[0][1],mpo[0][2],mpo[0][3])
                psi_conj = einsum('apb,bqc,crd,dse->pqrs',conj(mps[state][0]),conj(mps[state][1]),conj(mps[state][2]),conj(mps[state][3]))
                E[state] = einsum('pqrs,PpQqRrSs,PQRS',psi,op,psi_conj)
            return E
        # Calc initial Energy
        E1 = calc_energy(mpsL,mpo)
        mps = load_mps_list(mpsL)
        E1_ = calc_energy(mps,mpo,load_mps=False)
        E1__ = calc_energy(mps,mpo,load_mps=False)
        # Move gauge using renormalization
        site = N-1
        (mps0,) = retrieve_tensors(site-1,mpsList=mpsL)
        (mps1,) = retrieve_tensors(site,mpsList=mpsL)
        mps0_ = copy.deepcopy(mps0)
        mps1_ = copy.deepcopy(mps1)
        mps0,mps1,EE,EEs,wgt = renormalize_left(mps0,mps1,state_avg=True)
        save_tensors(site-1,mpsList=mpsL,mps=mps0)
        save_tensors(site,mpsList=mpsL,mps=mps1)
        # Check to make sure contracted states and local energies are the same
        for state in range(nState):
            finalState = einsum('apb,bqc->apqc',conj(mps0[state]),conj(mps1[state]))
            initState = einsum('apb,bqc->apqc',conj(mps0_[state]),conj(mps1_[state]))
            mpoComb = einsum('lPpm,mQqn->lPpQqn',mpo[0][site-1],mpo[0][site])
            initLocalE = einsum('apqc,lPpQqn,APQC->AlaCnc',initState,mpoComb,conj(initState))
            finalLocalE = einsum('apqc,lPpQqn,APQC->AlaCnc',finalState,mpoComb,conj(finalState))
            self.assertTrue(summ(abss(finalState-initState)) < 1e-10)
            self.assertTrue(summ(abss(initLocalE-initLocalE)) < 1e-10)
        # Check to make sure we get the same energies from resulting states
        for state in range(nState):
            mps[state][site-1] = mps0[state]
            mps[state][site]   = mps1[state]
        # Calc Energy Again
        E2 = calc_energy(mps,mpo,load_mps=False)
        # Check Energies
        self.assertTrue(summ(abss(E1[0]-E2[0])) < 1e-10)
        self.assertTrue(summ(abss(E1[1]-E2[1])) < 1e-10)
        self.assertTrue(summ(abss(E1[2]-E2[2])) < 1e-10)
        self.assertTrue(summ(abss(E1[3]-E2[3])) < 1e-10)
        # Check for correct canonical form
        for state in range(nState):
            eye_check = einsum('apb,cpb->ac',mps1[state],conj(mps1[state]))
            eye_actual= eye(eye_check.shape[0])
            self.assertTrue(summ(abss(eye_check-eye_actual)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)


if __name__ == "__main__":
    unittest.main()
