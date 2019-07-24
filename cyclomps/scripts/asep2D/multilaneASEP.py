from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.mpo.asep2D import return_mpo
from numpy import complex as complex_

# System Size
Ny = 5
Nx = 5
mbd = [10,100,500,1000,5000,10000,50000,100000]

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
min_iter = 3 
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


# Set Calculation Parameters
p = 0.1 
s0 = -0.5
sF = 0.5
ds0 = 0.01
alg = 'davidson'
leftState = False

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
