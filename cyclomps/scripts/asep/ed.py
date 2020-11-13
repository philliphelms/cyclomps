from cyclomps.tools.utils import *
from cyclomps.algs.ed import ed
from cyclomps.mpo.asep import return_mpo,curr_mpo
from cyclomps.mpo.ops import *
from cyclomps.tools.mpo_tools import *

# System Size
N = 2

# Hamiltonian Parameters
alpha = 0.1
gamma = 0.2 
p = 0.3
q = 0.4
beta = 0.5 
delta = 0.6 
s = 0.

# Set up mpo
hamParams = (alpha,gamma,p,q,beta,delta,s)
mpo = return_mpo(N,hamParams)

# Run ED
E0,vl,vr = ed(mpo,left=True)
vr0 = vr[:,0]
vl0 = vl[:,0]
print('E0 = {}'.format(E0[0].real))

# Print full state if wanted
#for i in range(len(vr0)):
#    print('{} {}'.format(vr0[i].real,vr0[i].imag))

# Evaluate helpful observables
# Density
density = zeros((N))
for i in range(N):
    rho_mpo = rho_mpo = [array([[I]])]*N
    rho_mpo[i] = array([[n]])
    rho_mpo = [rho_mpo]
    rho_mpo = reorder_bonds(rho_mpo)
    rho = mpo2mat(rho_mpo)
    density[i] = dot(vl0.conj(),dot(rho,vr0)).real
    print('Density({}) = {}'.format(i,density[i]))
print('Density_tot = {}'.format(summ(density)))
# Current
xcurr = 0.
for i in ['left'] + range(N-1) + ['right']:
    cmpo = curr_mpo(N,hamParams,singleBond=True,bond=i)
    curr = mpo2mat(cmpo)
    curr = dot(vl0.conj(),dot(curr,vr0)).real
    print('Horizontal Current({}) = {}'.format(i,curr))
    xcurr += curr
print('total Horizontal Current = {}'.format(xcurr))
# Current total
cmpo = curr_mpo(N,hamParams)
curr = mpo2mat(cmpo)
curr = dot(vl0.conj(),dot(curr,vr0)).real
print('Actual Total current = {}'.format(curr))
