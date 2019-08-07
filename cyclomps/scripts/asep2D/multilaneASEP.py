from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.mpo.asep2D import return_mpo
from numpy import complex as complex_
from sys import argv

# System Size
Ny = int(argv[1])
Nx = int(argv[2])
mbd = []
for i in range(3,len(argv)):
    mbd.append(int(argv[i]))
print(mbd)

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
alg = 'davidson'
tol = 1e-5
max_iter = 5
min_iter = 1
mps_dir = 'asep2D_mps'
env_dir = 'asep2D_env'
nStates = 2
fixed_bd = True
state_avg = True
orthonormalize = False
end_gauge = 0 
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

E0 = dmrg(mpo,
          alg=alg,
          dtype=complex_,
          mbd =mbd,
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
          left=left)
