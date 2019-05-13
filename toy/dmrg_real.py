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
maxBondDim = 20
##############################################

# Create MPS #################################
# PH - Make Isometries, Center Site
print('Generating MPS')
M = []
for i in range(int(N/2)):
    tmp = np.random.rand(2,
                         min(2**(i),maxBondDim),
                         min(2**(i+1),maxBondDim))
    M.append(ctf.from_nparray(tmp))
for i in range(int(N/2))[::-1]:
    tmp = np.random.rand(2,
                         min(2**(i+1),maxBondDim),
                         min(2**i,maxBondDim))
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
    M[i-1] = ctf.einsum('klj,ji,i->kli',M[i-1],U,s)
##############################################

# Create Environment #########################
# Allocate empty environment
F = []
tmp = np.array([[[1.]]])
F.append(ctf.from_nparray(tmp))
print(F[0])
for i in range(int(N/2)):
    tmp = np.zeros((min(2**(i+1),maxBondDim),4,min(2**(i+1),maxBondDim)))
    F.append(ctf.from_nparray(tmp))
for i in range(int(N/2)-1,0,-1):
    tmp = np.zeros((min(2**(i),maxBondDim),4,min(2**i,maxBondDim)))
    F.append(ctf.from_nparray(tmp))
tmp = np.array([[[1.]]])
F.append(ctf.from_nparray(tmp))
# Calculate initial environment
for i in range(int(N)-1,0,-1):
    tmp = np.einsum('eaf,cdf->eacd',M[i],F[i+1])
    tmp = np.einsum('ydbe,eacd->ybac',W[i],tmp)
    F[i] = np.einsum('bxc,ybac->xya',ctf.conj(M[i]),tmp)
##############################################

# Optimization Sweeps ########################


"""

# Optimization Sweeps ########################
converged = False
iterCnt = 0
E_prev = 0
while not converged:
# Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    for i in range(N-1):
        # Try to calculate diagonal:
        mpo_diag = np.einsum('abnn->anb',W[i])
        print(F[i+3].shape)
        l_diag = np.einsum('lal->la',F[i])
        r_diag = np.einsum('rbr->rb',F[i+1])
        scr = np.einsum('la,anb->lnb',l_diag,mpo_diag)
        print(np.einsum('lnb,rb->lnr',scr,r_diag))
        H = np.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        print(H)
        u,v = np.linalg.eig(H)
        # select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        v = v[:,max_ind]
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = np.reshape(v,(n1,n2,n3))
        # Right Normalize
        M_reshape = np.reshape(M[i],(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M[i] = np.reshape(U,(n1,n2,n3))
        M[i+1] = np.einsum('i,ij,kjl->kil',s,V,M[i+1])
        # Update F
        F[i+1] = np.einsum('jlp,ijk,lmin,npq->kmq',F[i],np.conj(M[i]),W[i],M[i])
# Left Sweep -----------------------------
    print('Left Sweep {}'.format(iterCnt))
    for i in range(N-1,0,-1):
        H = np.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        u,v = np.linalg.eig(H)
        # select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        v = v[:,max_ind]
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = np.reshape(v,(n1,n2,n3))
        # Right Normalize 
        M_reshape = np.swapaxes(M[i],0,1)
        M_reshape = np.reshape(M_reshape,(n2,n1*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n2,n1,n3))
        M[i] = np.swapaxes(M_reshape,0,1)
        M[i-1] = np.einsum('klj,ji,i->kli',M[i-1],U,s)
        # Update F
        F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[i]),W[i],M[i],F[i+1])
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
##############################################
"""
