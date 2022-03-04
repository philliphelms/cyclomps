from cyclomps.tools.utils import *
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.algs.ed import ed
from numpy import complex_
from cyclomps.mpo.dasep import get_dasep_mpo
from sys import argv

#####################################
# Create the MPO
#####################################
N = int(argv[1])
draw = False
periodic = False

# System parameters
params = {'wA': 0.01,   # Particle Addition
          'wD': 0.09,   # Particle Removal
          'kA': 1e-4,   # Defect Addition
          'kD': 9e-4,   # Defect Removal
          'wmA': 0.01,  # Particle addition to defect site
          'wmD': 0.09,  # Particle removal from defect site
          'kmA': 1e-4,  # Defect addition from site with particle
          'kmD': 9e-4,  # Defect removal from site with particle
          'v': 1.0,     # Forward hopping rate
          'vmp': 0.1,   # Forward hopping rate into defect
          'vmm': 0.1,   # Forward hopping rate out of defect
          'vm': 0.1}    # Forward hopping rate from defect to another defect

# Generate the mpo
mpo = get_dasep_mpo(N, params, periodic=periodic)

#####################################
# Run the dmrg calculation
#####################################
# Do an exact diagonalization (if wanted)
if N <= 5:
    u, v = ed(mpo)
    print(u)

#####################################
# Run the dmrg calculation
#####################################
E0, mps = dmrg(mpo,
               alg='davidson',
               dtype=complex_,
               mbd=[10, 100],
               tol=1e-5,
               max_iter = 10,
               min_iter = 1,
               nStates = 2,
               fixed_bd = True,
               state_avg = True)
