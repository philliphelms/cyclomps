from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from numpy import float_
from cyclomps.mpo.ising import return_mpo
import cProfile
import pstats

# System Size
N = 50

# Hamiltonian Parameters
h = 0.1

# Maximum bond dimension
mbdVec = [5,10,100,1000,2500,5000,7500,10000,20000,30000,40000,50000,75000,100000]

# Specify number of processors
n = 56

# Calculation Settings
alg = "'davidsonh'"
tol = 1e-5
max_iter = 1
min_iter = 0
mps_dir = "'ising_mps'"
env_dir = "'ising_env'"
nStates = 2
fixed_bd = True
state_avg = True
orthonormalize = False
end_gauge = 0
left = False

# Set up mpo
mpo = return_mpo(N,[h])

for mbdind,mbd in enumerate(mbdVec):
    # Run DMRG
    cProfile.run("E0 = dmrg(mpo,alg="+alg+",dtype=float_,mbd = "+str(mbd)+",tol="+str(tol)+",max_iter="+str(max_iter)+",min_iter="+str(min_iter)+",mps_subdir="+mps_dir+",env_subdir="+env_dir+",nStates="+str(nStates)+",fixed_bd="+str(fixed_bd)+",state_avg="+str(state_avg)+",orthonormalize="+str(orthonormalize)+",end_gauge="+str(end_gauge)+",left="+str(left)+")",'n'+str(n)+'_dmrg_stats_'+str(mbd))
    try:
        if RANK == 0:
            p = pstats.Stats('n'+str(n)+'_dmrg_stats_'+str(mbd))
    except:
        pass
