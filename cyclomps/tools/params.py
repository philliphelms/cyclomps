import os, sys
from mpi4py import MPI

# MPI Global Variables
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# Temporary directories for calculation
TMPDIR = os.environ.get('TMPDIR','.')
TMPDIR = os.environ.get('CYCLOMPS_TMPDIR',TMPDIR)

# Specify new directory for this calculation
if RANK == 0:
    ind = 0
    created = False
    while not created:
        CALCDIR = TMPDIR+'/cyclomps_calc'+str(ind)
        if not os.path.exists(CALCDIR):
            os.mkdir(CALCDIR)
            created = True
        ind += 1

# Use ctf or numpy
USE_CTF = True
USE_SPARSE = False

# Printing Global Variables
VERBOSE = 4
VERBOSE_TIME = 0
VERBOSE_MEM = 0
OUTPUT_DIGITS = 5
OUTPUT_COLS = 5

# Eigenproblem parameters
DAVIDSON_TOL = 1e-10
DAVIDSON_MAX_ITER = 100
USE_PRECOND = False
ARNOLDI_TOL = 1e-8
ARNOLDI_MAX_ITER = 100

# Memory Global Variables
import psutil
_,av,_,_,_,_,_,_,_,_,_ = psutil.virtual_memory()
MAX_MEMORY = av
