"""
Simple Time Evolving Block Decimation Algorithm Implementation

Author: Phillip Helms <phelms@caltech.edu>
Date: August 2019

"""
from cyclomps.tools.mpo_tools import *
from cyclomps.tools.mps_tools import *
import scipy.linalg as sla
from numpy import float_

def tebd_step(ten1,ten2,U,right=True,normalize=True):
    """
    """
    # Contract tensors with time evolution operator
    theta = einsum('aPb,bQc,PQpq->apqc',ten1,ten2,U)

    # Split resulting tensor
    U,S,V,EE,_ = svd_ten(theta,2,mbd=ten1.shape[2],return_wgt=False)
    if normalize: S /= sqrt(summ(S*S))

    # Get new tensors
    if right:
        ten1 = U
        ten2 = einsum('i,ijk->ijk',S,V)
    else:
        ten1 = einsum('ijk,k->ijk',U,S)
        ten2 = V

    # Return results
    return ten1,ten2,EE

def tebd_sweep_right(mps,ops,dt):
    """
    """
    N = len(mps[0])

    # Loop over sites
    for site in range(N-1):
        
        # Load two relevant tensors
        ten1 = mps_load_ten(mps,0,site)
        ten2 = mps_load_ten(mps,0,site+1)

        # Take the exponential of the hamiltonian operator
        (n1,n2,n3,n4) = ops[site].shape
        U = reshape(ops[site],(n1*n2,n3*n4))
        U = expm(U,a=dt)
        U = reshape(U,(n1,n2,n3,n4))

        # Do a single tebd step
        ten1,ten2,EE = tebd_step(ten1,ten2,U)

        # Save resulting tensors
        save_ten(ten1,mps[0][site]['fname'])
        save_ten(ten2,mps[0][site+1]['fname'])

    return 0,mps

def tebd_sweep_left(mps,ops,dt):
    """
    """
    N = len(mps[0])

    # Loop over sites
    for site in reversed(range(N-1)):
        
        # Load two relevant tensors
        ten1 = mps_load_ten(mps,0,site)
        ten2 = mps_load_ten(mps,0,site+1)

        # Take the exponential of the hamiltonian operator
        (n1,n2,n3,n4) = ops[site].shape
        U = reshape(ops[site],(n1*n2,n3*n4))
        U = expm(U,a=dt)
        U = reshape(U,(n1,n2,n3,n4))

        # Do a single tebd step
        ten1,ten2,EE = tebd_step(ten1,ten2,U,right=False)

        # Save resulting tensors
        save_ten(ten1,mps[0][site]['fname'])
        save_ten(ten2,mps[0][site+1]['fname'])

    return 0,mps

def tebd(ops,d=2,D=10,
         step_size=0.2,n_step=5,conv_tol=1e-8,
         H=None):
    """
    """
    t0 = time.time()
    mpiprint(0,'\n\nStarting TEBD Calculation')
    mpiprint(0,'#'*50)

    # Ensure the optimization parameters, namely the
    # bond dimension, trotter step size, and number
    # of trotter steps are compatable.
    if (not hasattr(D,'__len__')):
        D = [D]
        n_step = [n_step]
        step_size = [step_size]
        conv_tol = [conv_tol]

    # Get system size
    N = len(ops)+1

    # Create a random MPS
    mps = create_mps_list([d]*N,D[0],1,dtype=float_,fixed_bd=True)

    # Make it right canonical
    mps = make_mps_list_right(mps)

    # Calc initial energy
    if H is not None:
        Eo   = contract(mps=mps,mpo=H,state=0,gSite=0)
        norm = contract(mps=mps,mpo=None,state=0,gSite=0)
        E = Eo/norm
    else: E = 0
    print('E_0    = {}/{} = {}'.format(Eo,norm,E))

    # Loop over all (bond dims / step sizes / n_steps)
    for i in range(len(D)):
        print('\nNew Parameters = D({}),dt({})'.format(D[i],step_size[i]))

        # Do a tebd evolution for a given step size
        for step_cnt in range(n_step[i]):

            # Do a single tebd step
            E,mps = tebd_sweep_right(mps,ops,step_size[i])
            E,mps = tebd_sweep_left (mps,ops,step_size[i])

            # Check for Energy convergence
            if H is not None:
                Eo   = contract(mps=mps,mpo=H,state=0,gSite=0)
                norm = contract(mps=mps,mpo=None,state=0,gSite=0)
                E = Eo/norm
            else: E = 0
            print('Energy = {}/{} = {}'.format(Eo,norm,E))

        # Increase MBD if needed
        if (len(D)-1 > i) and (D[i+1] > D[i]):
            mps = increase_bond_dim(mps,D[i+1],fixed_bd=True,noise=1e-5)
    
    # Return resulting energy
    return E

    
if __name__ == "__main__":
    from cyclomps.mpo.tasep import return_mpo,return_tebd_ops
    from cyclomps.algs.dmrg1 import dmrg
    N = 30
    a = 0.5
    b = 0.5
    D = 10
    sVec = [-1.,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,-0.01,0.,0.01,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for sind,s in enumerate(sVec):
        # Get operators
        ops = return_tebd_ops(N,(a,b,s))
        ham = return_mpo(N,(a,b,s))
        # Start by trying DMRG
        E0,mps = dmrg(ham,
                      mbd = D,
                      max_iter = 5,
                      nStates = 1,
                      fixed_bd = True,
                      return_state = True)
        # Try again with TEBD
        E = tebd(ops,
                 H = ham,
                 D        =[  D,  D,  D,  D,  D,  D,  D,  D,  D],
                 step_size=[ 1.,0.9,0.7,0.6,0.5,0.4,0.3,0.2,0.1],
                 n_step   =[ 10, 10, 10, 10, 10, 10, 10, 10, 10],
                 conv_tol=1e-8)
        print(s,E)
