from cyclomps.tools.utils import *
from cyclomps.tools.mps_tools import contract
from cyclomps.algs.ed import ed
from cyclomps.mpo.asep2D import return_mpo
from cyclomps.mpo.asep2D import single_bond_energy
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex as complex_
from sys import argv

# System Size
Ny = int(argv[1])
Nx = int(argv[2])

# Hamiltonian Parameters
jr = 1.
jl = 0.
ju = 0.
jd = 0.
cr = 0.35
cl = 0.
cu = 0.
cd = 0.
dr = 2./3.
dl = 0.
du = 0.
dd = 0.
sx = -0.1
sy = 0.

# Calculate Settings
left = False

# Set up MPO
periodicy = False
periodicx = False
hamParams = array([jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy])
mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)

# Run diagonalization
E0,vl,vr = ed(mpo,left=True)
print(E0[0],E0[-1])

# Run dmrg
E,mps = dmrg(mpo,mbd=20,nStates=1,return_state=True)

# Evaluate Local Energies
# vertical interactions
print('\nVertical Energies')
for y in ['bottom']+[i for i in range(Ny-1)]+['top']:
    for x in range(Nx):
        mpo_tmp = single_bond_energy((Nx,Ny),hamParams,x,y,'vert')
        Etmp = contract(mps=mps,mpo=mpo_tmp)
        print('E_vert({},{}) = {}'.format(x,y,Etmp))
# Horizontal interactions
print('\nHorizontal Energies')
print(['left']+[i for i in range(Nx-1)] + ['right'])
for x in ['left']+[i for i in range(Nx-1)] + ['right']:
    for y in range(Ny):
        mpo_tmp = single_bond_energy((Nx,Ny),hamParams,x,y,'horz')
        Etmp = contract(mps=mps,mpo=mpo_tmp)
        print('E_horz({},{}) = {}'.format(x,y,Etmp))


