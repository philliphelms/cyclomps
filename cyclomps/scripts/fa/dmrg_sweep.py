from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.mpo.fa import return_mpo
from cyclomps.tools.mps_tools import contract_config
from numpy import complex as complex_
from sys import argv
from numpy import logspace,linspace 

# System Size
N = int(argv[1])
mbd = []
for i in range(2,len(argv)):
    mbd.append(int(argv[i]))

# Hamiltonian Parameters
c = 0.5
sVec = linspace(0,0.2,30)

# Calculate Settings
hermitian = False # Use the hermitian version of the FA hamiltonian
alg = 'davidson'
tol = 1e-5
max_iter = 10
min_iter = 0 
mps_dir = 'fa_mps'
env_dir = 'fa_env'
nStates = 3
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
    print('s = {}'.format(s))
    hamParams = array([c,s,0.])
    mpo = return_mpo(N,hamParams,hermitian=hermitian)

    # Run diagonalization
    out = dmrg(mpo,
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

    # Collect output
    if left:
        E0  = out[0]
        E0l = out[1]
        mps = out[2]
        mpsl= out[3]
        env = out[4]
        envl= out[5]
    else:
        E0  = out[0]
        mps = out[1]
        env = out[2]

    # Just for fun, contract configuration sample
    #rand_config = [0 if np.random.random() > 0.5 else 1 for i in range(N)]
    #print(rand_config)
    #config_sample = contract_config(mps,
    #                                rand_config)
    #print('Configuration sample value = {}'.format(config_sample))

    # Save and print out energies
    E0_vec.append(real(E0[0]))
    E1_vec.append(real(E0[1]))
    for sind2 in range(len(E0_vec)):
        mpiprint(0,'{}\t{}\t{}'.format(sVec[sind2],E0_vec[sind2],E1_vec[sind2]))
    try:
        mbd = mbd[-1]
    except:
        pass
