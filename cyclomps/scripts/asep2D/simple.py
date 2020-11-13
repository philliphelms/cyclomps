from cyclomps.tools.utils import *
from cyclomps.tools.mps_tools import contract
from cyclomps.algs.ed import ed
from cyclomps.mpo.asep2D import return_mpo
from cyclomps.mpo.asep2D import single_bond_energy
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex as complex_
from sys import argv
import time

# System Size
Ny = 10
Nx = 10

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
sx = -0.1
sy = -0.5

# Calculate Settings
left = False

# Set up MPO
periodicy = False
periodicx = False
hamParams = array([jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy])
mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)

# Run diagonalization
#E0,vl,vr = ed(mpo,left=True)
#print(E0[0],E0[-1])

# Run dmrg
t0 = time.time()
E,mps = dmrg(mpo,mbd=2,nStates=1,return_state=True)
tf = time.time()
print(E,tf-t0)
