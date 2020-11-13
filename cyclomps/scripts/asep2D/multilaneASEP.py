from cyclomps.tools.utils import *
from cyclomps.tools.mps_tools import increase_bond_dim
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.mpo.asep2D import return_mpo
from numpy import complex as complex_, linspace, savez
from sys import argv

# System Size
Ny = int(argv[1])
Nx = int(argv[2])
mbd = []
for i in range(3,len(argv)):
    mbd.append(int(argv[i]))

# Hold sweep results
sxVec = linspace(-3,1,81)
syVec = linspace(-3,1,81)
E = zeros((len(mbd),len(sxVec),len(syVec)))

mps = None
for sxind in range(len(sxVec)):
    for syind in range(len(syVec)):
        # Hamiltonian Parameters
        bcs = 'open'
        jr = 0.9
        jl = 1.-jr
        ju = 0.9
        jd = 1.-ju
        cr = 0.5
        cl = 0.5
        cu = 0.5
        cd = 0.5
        dr = 0.5
        dl = 0.5
        du = 0.5
        dd = 0.5
        sx = sxVec[sxind]
        sy = syVec[syind]
        hamParams = array([jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy])

        # Calculate Settings
        alg = 'exact'
        tol = 1e-5
        max_iter = 3
        min_iter = 0
        mps_dir = 'asep2D_mps'
        env_dir = 'asep2D_env'
        nStates = 1
        fixed_bd = True
        state_avg = True
        orthonormalize = False
        end_gauge = 0 
        left = False


        # Set up MPO
        periodicy = False
        periodicx = False
        mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
        mps = None

        for Dind in range(len(mbd)):
            res = dmrg(mpo,
                      mps=mps,
                      alg=alg,
                      dtype=complex_,
                      mbd =mbd[Dind],
                      tol=tol,
                      max_iter=max_iter,
                      min_iter=min_iter,
                      mps_subdir=mps_dir,
                      env_subdir=env_dir, 
                      nStates=nStates,
                      fixed_bd=fixed_bd,
                      state_avg=state_avg,
                      orthonormalize=orthonormalize,
                      end_gauge=end_gauge,
                      return_state=True,
                      left=left)
            E[Dind,sxind,syind] = res[0][0]
            mps = res[1]
            if Dind < len(mbd)-1:
                mps = increase_bond_dim(mps,mbd[Dind],fixed_bd=fixed_bd)

        print(E[:,sxind,syind])

        # Save results
        savez('_data',E=E)
