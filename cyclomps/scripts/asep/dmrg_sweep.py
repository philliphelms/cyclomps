from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
#from numpy import complex_
from cyclomps.mpo.asep import return_mpo
#from sys import argv
#from numpy import logspace,linspace 

# System Size
N = 10
mbd = 10

# Hamiltonian Parameters
alpha = 0.5 
gamma = 0.5 
p = 0.1 
q = 1.-p
beta = 0.5 
delta = 0.5 
s = -1.

# Calculate Settings
alg = 'davidson'
tol = 1e-5
max_iter = 5
min_iter = 1 
mps_dir = 'fa_mps'
env_dir = 'fa_env'
nStates = 2
fixed_bd = True
state_avg = True
orthonormalize = False
end_gauge = 0 
left = False

# Somewhere to store resulting E
E = []
mps = None
env = None
# Loop over s

#s = sVec[0]
# Set up mpo
hamParams = (alpha,gamma,p,q,beta,delta,s)
mpo = return_mpo(N,hamParams)
print('Got the mpo')

# Run diagonalization
E0 = dmrg(mpo,
                  alg=alg,
                  mps=mps,
                  env=env,
                  #return_state=True,
                  #return_env=True,
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
"""
for sind,s in enumerate(sVec):
    
    # Set up mpo
    hamParams = array([alpha,gamma,p,q,beta,delta,s])
    mpo = return_mpo(N,hamParams)

    # Run diagonalization
    E0,mps,env = dmrg(mpo,
                      alg=alg,
                      mps=mps,
                      env=env,
                      return_state=True,
                      return_env=True,
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
    E.append(E0[1])
    for sind2 in range(len(E)):
        print('{}\t{}\t{}'.format(sVec[sind2],real(E[sind2]),imag(E[sind2])))
"""
