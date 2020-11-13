from cyclomps.mpo.rydberg import return_mpo
from cyclomps.tools.mpo_tools import *
from cyclomps.tools.mps_tools import create_simple_state
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.algs.ed import ed

np.set_printoptions(linewidth=1000)

# System Size 
N = 4
Omega0 = 1.
Delta0 = 0.5
V0 = 1.
Omegaf = 0.01
Deltaf = 0.5
Vf = 1.
dt = 0.001
tf = 10.
imaginary_time = False

# Find ground state using the initial system
hamParams = (Omega0,Delta0,V0)
mpo = return_mpo(N,hamParams)
print(mpo[0][0])
print(mpo[0][1].shape)
print(mpo[0][2].shape)
print(mpo[0][3].shape)
H = mpo2mat(mpo)
print(H)
e,v = ed(mpo)
v = v[:,0]
print('Ground State Energy = {}'.format(e[0]))
for i in range(len(e)):
    print(e[i].real)

# Find goal ground state energy
hamParams = (Omegaf,Deltaf,Vf)
mpo2 = return_mpo(N,hamParams)
H2 = mpo2mat(mpo2)
e2,v2 = ed(mpo2)
print('New Ground State Energy = {}'.format(e2[0]))

# Get time evolution operator
if imaginary_time:
    eH2 = expm(dt*H2)
else:
    eH2 = expm(-1.j*dt*H2)

# Do time evolution
psi = v
psi_reshape2 = reshape(psi,(2**int(N/2),2**int(N/2)))
(U,S,V) = svd(psi_reshape2)
EE = -dot(S**2,log2(S**2))
# allocate arrays to store results
tvec = []
EEvec = []
E1vec = []
E2vec = []
for step in range(int(tf/dt)):
    # Store results
    tvec.append(dt*step)
    E1vec.append(real(dot(psi,dot(H,conj(psi)))))
    E2vec.append(real(dot(psi,dot(H2,conj(psi)))))
    EEvec.append(EE)

    # Do time step
    psi = dot(eH2,psi)
    psi /= sqrt(dot(psi,conj(psi)))

    # Check Observables -------------------
    psi_reshape = reshape(psi,(2,)*N)
    # Measure entropy
    psi_reshape2 = reshape(psi,(2**int(N/2),2**int(N/2)))
    (U,S,V) = svd(psi_reshape2)
    EE = -dot(S**2,log2(S**2))

if True:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    plt.plot(tvec,EEvec)
    plt.savefig('EE.pdf')


