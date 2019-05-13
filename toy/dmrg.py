import ctf
import numpy as np

######## Inputs ##############################
# SEP Model
N = 20
alpha = 0.35 # In at left
beta = 2./3. # Exit at right
s = -1.      # Exponential weighting
gamma = 0.   # Exit at left
delta = 0.   # In at right
p = 1.       # Jump right
q = 0.       # Jump Left
# Optimization
tol = 1e-5
maxIter = 10
maxBondDim = 10
useCTF = True
##############################################

# Create MPS #################################
# PH - Make Isometries, Center Site
print('Generating MPS')
M = []
for i in range(int(N/2)):
    tmp = np.random.rand(2,
                         min(2**(i),maxBondDim),
                         min(2**(i+1),maxBondDim))\
                         +0.j
    M.append(ctf.from_nparray(tmp))
for i in range(int(N/2))[::-1]:
    tmp = np.random.rand(2,
                         min(2**(i+1),maxBondDim),
                         min(2**i,maxBondDim))\
                         +0.j
    M.append(ctf.from_nparray(tmp))
##############################################

# Create MPO #################################
print('Generating MPO')
# Simple Operators
Sp = np.array([[0.,1.],[0.,0.]])
Sm = np.array([[0.,0.],[1.,0.]])
n = np.array([[0.,0.],[0.,1.]])
v = np.array([[1.,0.],[0.,0.]])
I = np.array([[1.,0.],[0.,1.]])
z = np.array([[0.,0.],[0.,0.]])
# List to hold MPOs
W = []
# First Site
site_0 = ctf.astensor([[alpha*(np.exp(-s)*Sm-v),
                        np.exp(-s)*Sp,
                        -n,
                        I                       ]])
W.append(site_0)
# Central Sites
for i in range(N-2):
    site_i = ctf.astensor([[I,             z,  z, z],
                           [Sm,            z,  z, z],
                           [v,             z,  z, z],
                           [z, np.exp(-s)*Sp, -n, I]])
    W.append(site_i)
# Last Site
site_N = ctf.astensor([[I                     ],
                       [Sm                    ],
                       [v                     ],
                       [beta*(np.exp(-s)*Sp-n)]])
W.append(site_N)
##############################################

# Canonicalize MPS ###########################
for i in range(int(N)-1,0,-1):
    M_reshape = ctf.transpose(M[i],axes=[1,0,2])
    (n1,n2,n3) = M_reshape.shape
    M_reshape = M_reshape.reshape(n1,n2*n3)
    (U,S,V) = ctf.svd(M_reshape)
    M_reshape = V.reshape(n1,n2,n3)
    M[i] = ctf.transpose(M_reshape,axes=[1,0,2])
    M[i-1] = ctf.einsum('klj,ji,i->kli',M[i-1],U,S)
##############################################

# Create Environment #########################
print('Generating Environment')
# Allocate empty environment
F = []
tmp = np.array([[[1.]]])+0.j
F.append(ctf.from_nparray(tmp))
for i in range(int(N/2)):
    tmp = np.zeros((min(2**(i+1),maxBondDim),4,min(2**(i+1),maxBondDim)))+0.j
    F.append(ctf.from_nparray(tmp))
for i in range(int(N/2)-1,0,-1):
    tmp = np.zeros((min(2**(i),maxBondDim),4,min(2**i,maxBondDim)))+0.j
    F.append(ctf.from_nparray(tmp))
tmp = np.array([[[1.]]])+0.j
F.append(ctf.from_nparray(tmp))
# Calculate initial environment
for i in range(int(N)-1,0,-1):
    tmp = ctf.einsum('eaf,cdf->eacd',M[i],F[i+1])
    tmp = ctf.einsum('ydbe,eacd->ybac',W[i],tmp)
    F[i] = ctf.einsum('bxc,ybac->xya',ctf.conj(M[i]),tmp)
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
E_prev = 0
while not converged:
    # Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    for i in range(N-1):
        H = ctf.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = ctf.reshape(H,(n1*n2*n3,n4*n5*n6))
        u,v = np.linalg.eig(ctf.to_nparray(H))
        # Select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        v = ctf.from_nparray(v[:,max_ind])
        print('\tEnergy at site {} = {}'.format(i,E))
        M[i] = ctf.reshape(v,(n1,n2,n3))
        # Right Normalize
        M_reshape = ctf.reshape(M[i],(n1*n2,n3))
        (U,S,V) = ctf.svd(M_reshape)
        M[i] = ctf.reshape(U,(n1,n2,n3))
        M[i+1] = ctf.einsum('i,ij,kjl->kil',S,V,M[i+1])
        # Update F
        F[i+1] = ctf.einsum('jlp,ijk,lmin,npq->kmq',F[i],ctf.conj(M[i]),W[i],M[i])
    # Left Sweep ------------------------------
    print('Left Sweep {}'.format(iterCnt))
    for i in range(N-1,0,-1):
        H = ctf.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = ctf.reshape(H,(n1*n2*n3,n4*n5*n6))
        u,v = np.linalg.eig(ctf.to_nparray(H))
        # Select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        v = ctf.from_nparray(v[:,max_ind])
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = ctf.reshape(v,(n1,n2,n3))
        # Right Normalize 
        M_reshape = ctf.transpose(M[i],(1,0,2))
        M_reshape = ctf.reshape(M_reshape,(n2,n1*n3))
        (U,S,V) = ctf.svd(M_reshape)
        M_reshape = ctf.reshape(V,(n2,n1,n3))
        M[i] = ctf.transpose(M_reshape,(1,0,2))
        M[i-1] = ctf.einsum('klj,ji,i->kli',M[i-1],U,S)
        # Update F
        F[i] = ctf.einsum('bxc,ydbe,eaf,cdf->xya',ctf.conj(M[i]),W[i],M[i],F[i+1])
    # Convergence Test -----------------------
    if np.abs(E-E_prev) < tol:
        print('#'*75+'\nConverged at E = {}'.format(E)+'\n'+'#'*75)
        converged = True
    elif iterCnt > maxIter:
        print('Convergence not acheived')
        converged = True
    else:
        iterCnt += 1
        E_prev = E
