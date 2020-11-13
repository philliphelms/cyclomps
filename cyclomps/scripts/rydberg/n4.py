from cyclomps.tools.mpo_tools import *
from cyclomps.tools.utils import *
from cyclomps.algs.ed import ed
#from numpy import exp

tau = 0.1
tf = 1.


# Create the MPO
mpo = []
# Site 1
ten = zeros((1,4,2,2))
ten[0,0,0,0] = 1.
ten[0,0,1,1] = 1.
ten[0,1,0,0] = -0.001
ten[0,1,1,0] = -0.05
ten[0,1,0,1] = -0.05
ten[0,2,1,0] = 0.5
ten[0,2,0,1] = 0.5
ten[0,3,0,0] = 1.
mpo.append(ten)
# Site 2
ten = zeros((4,5,2,2))
ten[0,0,0,0] = 1.
ten[0,0,1,1] = 1.
ten[0,1,0,0] = -0.001
ten[0,1,1,0] = -0.05
ten[0,1,0,1] = -0.05
ten[1,1,0,0] = 1.
ten[1,1,1,1] = 1.
ten[3,1,0,0] = 1.
ten[0,2,1,0] = 0.5
ten[0,2,0,1] = 0.5
ten[0,3,0,0] = 1.
ten[3,4,0,0] = 1.
ten[3,4,1,1] = 1.
mpo.append(ten)
# Site 3
ten = zeros((5,6,2,2))
ten[0,0,0,0] = 1.
ten[0,0,1,1] = 1.
ten[0,1,0,0] = -0.001
ten[0,1,1,0] = -0.05
ten[0,1,0,1] = -0.05
ten[1,1,0,0] = 1.
ten[1,1,1,1] = 1.
ten[3,1,0,0] = 1.
ten[4,1,0,0] = 0.0156250
ten[0,2,1,0] = 0.5
ten[0,2,0,1] = 0.5
ten[0,3,0,0] = 1.
ten[4,4,0,0] = 1.
ten[4,4,1,1] = 1.
ten[3,5,0,0] = 1.
ten[3,5,1,1] = 1.
mpo.append(ten)
# Site 4
ten = zeros((6,1,2,2))
ten[0,0,0,0] = -0.001
ten[0,0,1,0] = -0.05
ten[0,0,0,1] = -0.05
ten[1,0,0,0] = 1.
ten[1,0,1,1] = 1.
ten[3,0,0,0] = 1.
ten[4,0,0,0] = 0.0013717
ten[5,0,0,0] = 0.0156250
mpo.append(ten)
# Make the mpo a 'list of mpos'
mpo = [mpo]
# Reorder the bonds
mpo = reorder_bonds(mpo)

# Run DMRG On the result
# Run ED on the result
u,v = ed(mpo)
print(u)
v = v[:,-1]
u = u[-1]
print('Ground State Energy = {}'.format(u))

# Now do some time evolution
H = mpo2mat(mpo)
eH = exp(-1.j*tau*H)

# Create the exponentiated MPO independently as a check
empo = []
# Site 1
ten = zeros((1,2,2,2))
ten[0,0,0,0] = 1.0000100
ten[0,0,1,0] = 0.0005000
ten[0,0,0,1] = 0.0005000
ten[0,0,1,1] = 1.0000000
ten[0,1,0,0] = -0.01
empo.append(ten)
# Site 2
ten = zeros((2,3,2,2))
ten[0,0,0,0] = 1.0000100
ten[0,0,1,0] = 0.0005000
ten[0,0,0,1] = 0.0005000
ten[0,0,1,1] = 1.0
ten[0,2,0,0] = -0.01
ten[1,0,0,0] = 1.
ten[1,1,0,0] = 1
ten[1,1,1,1] = 1.
empo.append(ten)
# Site 3
ten = zeros((3,4,2,2))
ten[0,0,0,0] = 1.0000100
ten[0,0,1,0] = 0.0005000
ten[0,0,0,1] = 0.0005000
ten[0,0,1,1] = 1.000
ten[0,3,0,0] = -0.01
ten[1,0,0,0] = 0.0156250
ten[1,1,0,0] = 1.
ten[1,1,1,1] = 1.
ten[2,0,0,0] = 1.
ten[2,2,0,0] = 1.
ten[2,2,1,1] = 1.
empo.append(ten)
# Site 4
ten = zeros((4,1,2,2))
ten[0,0,0,0] = 1.0000100
ten[0,0,1,0] = 0.0005000
ten[0,0,0,1] = 0.0005000
ten[0,0,1,1] = 1.
ten[1,0,0,0] = 0.0013717
ten[2,0,0,0] = 0.0156250
ten[3,0,0,0] = 1.
empo.append(ten)
# Make the mpo a 'list of mpos'
empo = [empo]
empo = reorder_bonds(empo)
eH2 = mpo2mat(empo)
#print('Here it is')
#print(eH-eH2)
eH = eH2

# Reorder the bonds
mpo = reorder_bonds(mpo)
nStep = int(tf/tau+(1e-9*(tf/tau)))
for step in range(nStep):
    # Do time step
    v = dot(v,eH)
    # Normalize v
    v /= sqrt(dot(v,conj(v)))
    # Evaluate energy
    Energy = dot(v,dot(H,conj(v)))
    print('E(t={}) = {}'.format(step*tau,Energy))
