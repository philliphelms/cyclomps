from cyclomps.tools.utils import *
from cyclomps.tools.mps_tools import contract
from cyclomps.algs.ed import ed
from cyclomps.mpo.asep2D import return_mpo
from cyclomps.mpo.asep2D import single_bond_energy
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex as complex_,linspace
from sys import argv
import time

# System Size
Ny = int(argv[1])
Nx = int(argv[2])
mbd= int(argv[3])
nStates=2

sxvec = linspace(-0.5,0.5,50)[::-1]


# Loop over all x biases
mps = None
for i in range(len(sxvec)):
    # Hamiltonian Parameters
    jr = 0.9
    jl = 0.1
    ju = 0.9
    jd = 0.1
    cr = 0.5
    cl = 0.5
    cu = 0.5
    cd = 0.5
    dr = 0.5
    dl = 0.5
    du = 0.5
    dd = 0.5
    sx = sxvec[i]
    sy = -0.5

    # Set up MPO
    periodicy = False
    periodicx = False
    hamParams = array([jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy])
    mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)

    # Run dmrg
    E,mps = dmrg(mpo,
                 mbd=mbd,
                 nStates=2,
                 return_state=True,
                 left=False,
                 max_iter=5,
                 fixed_bd=True,
                 state_avg=True)
    print('results,{},{},{},{}'.format(sx,E[0].real,E[1].real,(E[0]-E[1]).real))
