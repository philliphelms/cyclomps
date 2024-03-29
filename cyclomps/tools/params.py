import os, sys
#from mpi4py import MPI

# MPI Global Variables
COMM = 0#MPI.COMM_WORLD
RANK = 0#COMM.Get_rank()
SIZE = 1#COMM.size
#print('Rank = {}'.format(RANK))

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
                created = True
            except:
                pass
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
VERBOSE_MEM = 0
OUTPUT_DIGITS = 5
OUTPUT_COLS = 5

# Eigenproblem parameters
DAVIDSON_TOL = 1e-16
DAVIDSON_MAX_ITER = 100
USE_PRECOND = False
ARNOLDI_TOL = 1e-8
ARNOLDI_MAX_ITER = 100

# Memory Global Variables
#import psutil
#_,av,_,_,_,_,_,_,_,_,_ = psutil.virtual_memory()
MAX_MEMORY = None
