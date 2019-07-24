from cyclomps.tools.utils import *
from cyclomps.algs.ed import ed
from cyclomps.mpo.asep2D import return_mpo
from numpy import complex as complex_
from sys import argv

# System Size
Ny = int(argv[1])
Nx = int(argv[2])

# Hamiltonian Parameters
bcs = 'open'
jr = 0.1
jl = 1.-jr
ju = 0.5
jd = 0.5
cr = 0.5
cl = 0.5
cu = 0.5
cd = 0.5
dr = 0.5
dl = 0.5
du = 0.5
dd = 0.5
sx = -0.5
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
