import os, sys
from mpi4py import MPI
import ctf


# MPI Global Variables
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size

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
            try:
                os.mkdir(CALCDIR)
            except:
                pass
            created = True
        ind += 1
    for ind in range(1,SIZE):
        COMM.send({'dir': CALCDIR}, dest=ind)
else:
    dic = COMM.recv(source=0)
    CALCDIR = dic['dir']

# Use ctf or numpy
USE_CTF = False
USE_SPARSE = False

# Printing Global Variables
VERBOSE = 3
VERBOSE_TIME = 0
VERBOSE_MEM = -1
OUTPUT_DIGITS = 5
OUTPUT_COLS = 5

# Eigenproblem parameters
DAVIDSON_TOL = 1e-16
DAVIDSON_MAX_ITER = 1000
USE_PRECOND = True
ARNOLDI_TOL = 1e-16
ARNOLDI_MAX_ITER = 1000

# Memory Global Variables
import psutil
_,av,_,_,_,_,_,_,_,_,_ = psutil.virtual_memory()
MAX_MEMORY = av
