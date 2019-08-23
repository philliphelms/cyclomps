from cyclomps.tools.utils import *
from cyclomps.algs.ed import ed
from cyclomps.mpo.asep2D import return_mpo
from numpy import complex as complex_
from sys import argv

# System Size
Ny = int(argv[1])
Nx = int(argv[2])

# Hamiltonian Parameters
bcs = 'closed'
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
sx = -1.
sy = 0.

# Calculate Settings
left = False

# Set up MPO
periodicy = False
periodicx = False
if bcs == 'open':
    hamParams = array([jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy])
elif bcs == 'closed':
    hamParams = array([jr,jl,ju,jd,cr,cl,0.,0.,dr,dl,0.,0.,sx,sy])
elif bcs == 'periodic':
    hamParams = array([jr,jl,ju,jd,cr,cl,0.,0.,dr,dl,0.,0.,sx,sy])
    periodicy = True
mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)

# Run diagonalization
if left:
    E0,vl,vr = ed(mpo,left=True)
else:
    E0,vr = ed(mpo,left=False)

print('E0 = {}'.format(E0))
