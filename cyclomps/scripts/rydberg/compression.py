from cyclomps.mpo.rydberg_compressed import return_mpo as compressed_mpo
from cyclomps.mpo.rydberg import return_mpo
from cyclomps.tools.mpo_tools import *
from cyclomps.tools.mps_tools import create_simple_state
from cyclomps.algs.dmrg1 import dmrg

np.set_printoptions(linewidth=1000)

# System Size
N = 100
mbd = 50

# Hamiltonian Parameters
Omega = 0.1 
Delta = 0.001
V     = 1.
hamParams = (Omega,Delta,V)

# Get the original MPO
mpo = return_mpo(N,hamParams)

# Get the compressed MPO
mpoc = compressed_mpo(N,hamParams)

# Compare hamiltonians
if N < 10:
    ham = mpo2mat(mpo)
    hamc= mpo2mat(mpoc)
    print('Hamiltonian Differences: {}'.format(summ(abss(ham-hamc))))

# Run DMRG with not compressed MPO
energy_gs,mps = dmrg(mpo,
                 mbd=10,
                 left=False,
                 nStates=1,
                 return_state=True)

# Run DMRG with compressed MPO
energy_gsc,mps = dmrg(mpoc,
                 mps=mps,
                 mbd=10,
                 left=False,
                 nStates=1,
                 return_state=True)

print('\n'+'='*100)
print('Energy (full MPO): {}'.format(energy_gs))
print('Energy (comp MPO): {}'.format(energy_gsc))
print('='*100)

# Figure out what happens as D decreases
Dvec = [100,10,9,8,7,6,5,4,3,2,1]
Evec = []
for Dind in range(len(Dvec)):
    D = Dvec[Dind]
    mpoc = compressed_mpo(N,hamParams,D=D)
    energy_gsc,mps = dmrg(mpoc,
                     mps=mps,
                     mbd=10,
                     left=False,
                     nStates=1,
                     return_state=True)
    Evec.append(energy_gsc[0])

for ind in range(len(Dvec)):
    print('{}\t{}'.format(Dvec[ind],real(Evec[ind])))

