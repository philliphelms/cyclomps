from cyclomps.tools.utils import *
from cyclomps.algs.ed import ed
from cyclomps.mpo.fa import return_mpo
from cyclomps.tools.mpo_tools import *
from numpy import complex as complex_
from sys import argv
from numpy import logspace,linspace 
np.set_printoptions(linewidth=1000)

# System Size
N = int(argv[1])

# Hamiltonian Parameters
c = 0.1
mu = 0
sVec = linspace(-0.5,0.5,20)
#sVec = logspace(-5,0,100)
#sVec = array([-0.5])

# Calculate Settings
left = False
hermitian = False # Use the hermitian version of the FA hamiltonian

# Loop over s
for sind,s in enumerate(sVec):
    
    # Set up mpo
    hamParams = array([c,s,mu])
    mpo = return_mpo(N,hamParams,hermitian=hermitian)
    ham = mpo2mat(mpo)

    # Run diagonalization
    if left:
        E0,vl,vr = ed(mpo,left=True)
    else:
        E0,vr = ed(mpo,left=False)
        vr = vr[:,0]

    # Subtract out chemical potential from energy
    nState = len(vr)
    n_cnt_op = zeros((nState,nState))
    for state in range(len(vr)):
        bin_str = "{0:b}".format(state)
        for bit in range(len(bin_str)):
            if bin_str[bit] == "1":
                n_cnt_op[state,state] += 1

    # Meausre occupation
    total_occ = dot(vr,dot(n_cnt_op,conj(vr)))

    # Adjust because of chemical potential:
    E0 = E0

    # Print out results
    #print(mu)
    #print(s)
    #print(total_occ)
    for i in range(len(E0)):
        print(real(E0[i]))
    print('-'*50)
