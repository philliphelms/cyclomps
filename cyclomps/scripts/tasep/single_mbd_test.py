import unittest
from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from numpy import complex_
from cyclomps.mpo.tasep import return_mpo
import cProfile
import pstats
mpo = return_mpo(10,(0.5,0.5,1.))
mbd = 10
n = 28
cProfile.run("E0 = dmrg(mpo,alg='davidson',dtype=complex_,mbd = "+str(mbd)+",tol=1e-5,max_iter=2,min_iter=2,mps_subdir='tasep_mps',env_subdir='tasep_env',nStates=1,fixed_bd=True,state_avg=False,orthonormalize=False,end_gauge=0,left=False)",'n'+str(n)+'_dmrg_stats_'+str(mbd))
if RANK == 0:
    p = pstats.Stats('n'+str(n)+'_dmrg_stats_'+str(mbd))
