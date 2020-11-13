from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.mpo.fa import return_mpo
from numpy import complex as complex_
from sys import argv
from numpy import logspace,linspace 

# System Size
N = int(argv[1])
mbd = []
for i in range(2,len(argv)):
    mbd.append(int(argv[i]))

# Hamiltonian Parameters
c = 0.2
#sVec = logspace(-5,0,30)[::-1]
sVec = linspace(-1,1,30)
#sVec = [-0.02,-0.01,0.,0.1,0.2]

# Calculate Settings
hermitian = True # Use the hermitian version of the FA hamiltonian
alg = 'davidson'
tol = 1e-5
max_iter = 2
min_iter = 0 
mps_dir = 'fa_mps'
env_dir = 'fa_env'
nStates = 2
fixed_bd = True
state_avg = True
orthonormalize = False
end_gauge = 0 
left = False

# Somewhere to store resulting E
E0_vec = []
E1_vec = []
mps = None
env = None
# Loop over s
for sind,s in enumerate(sVec):
    
    # Set up mpo
    hamParams = array([c,s,0.])
    mpo = return_mpo(N,hamParams,hermitian=hermitian)

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
    E0_vec.append(real(E0[0]))
    E1_vec.append(real(E0[1]))
    for sind2 in range(len(E0_vec)):
        mpiprint(0,'{}\t{}\t{}'.format(sVec[sind2],E0_vec[sind2],E1_vec[sind2]))
    try:
        mbd = mbd[-1]
    except:
        pass
