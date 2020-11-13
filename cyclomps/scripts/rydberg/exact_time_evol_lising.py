from cyclomps.mpo.long_ising import return_mpo
from cyclomps.tools.mpo_tools import *
from cyclomps.tools.mps_tools import create_simple_state
from cyclomps.algs.dmrg1 import dmrg
from cyclomps.algs.ed import ed
from cyclomps.mpo.ops import X,Z
from numpy import argsort

np.set_printoptions(linewidth=1000)

####################################################
# Calculation Parameters
# System Size 
N = 2
# Init System Parameters
hz0 = 0.5
hx0 = 0.1
J0  = -1.
# Final System Parameters
hzf = 0.25
hxf = 0.1
Jf  = -1.
# Time Evolution Parameters
time_evolve = False
dt = 0.1
tf = 30.
imaginary_time = False

####################################################
# Ground State
hamParams = (J0,hx0,hz0)
mpo = return_mpo(N,hamParams)
H = mpo2mat(mpo)
print(H)
e,v = ed(mpo)
v = v[:,0]
psi = v
inds = argsort(abs(v))[::-1]
for state in range(len(psi)):
    print(('{0:0'+str(N)+'b}').format(inds[state])+' {}'.format(psi[inds[state]]))

####################################################
# Quench Hamiltonian
hamParams = (Jf,hxf,hzf)
mpo2 = return_mpo(N,hamParams)
H2 = mpo2mat(mpo2)
e2,v2 = ed(mpo2)

# Get time evolution operator
if imaginary_time:
    eH2 = expm(dt*H2)
else:
    eH2 = expm(-1.j*dt*H2)

# Check Observables -------------------
# Measure entropy
psi_reshape = reshape(psi,(2**int(N/2),2**int(N/2)))
(U,S,V) = svd(psi_reshape)
EE = -dot(S**2,log2(S**2))
# Measure <Sxi>
psi_reshape = reshape(psi,(2,)*N)
Sx = zeros(N)
for site in range(N):
    string = 'abcdefghijklmnopqrstuvwxyz'
    str_in = string[:N]
    str_ot = string[:site]+string[site].upper()+string[site+1:N]
    str_op = string[site]+string[site].upper()
    str_ein= str_in+','+str_ot+','+str_op+'->'
    Sx[site] = einsum(str_ein,psi_reshape,conj(psi_reshape),X)
# measure <Sx_i,Sx_{i+1}>
SxSx = zeros(N-1)
for site in range(N-1):
    string = 'abcdefghijklmnopqrstuvwxyz'
    str_in = string[:N]
    str_ot = string[:site]+string[site:site+2].upper()+string[site+2:N]
    str_op1= string[site]+string[site].upper()
    str_op2= string[site+1]+string[site+1].upper()
    str_ein= str_in+','+str_ot+','+str_op1+','+str_op2+'->'
    SxSx[site] = einsum(str_ein,psi_reshape,conj(psi_reshape),X,X)
# measure <Sx1,Sx2,...,SxN>
string = 'abcdefghijklmnopqrstuvwxyz'
str_in_bra = string[:N]
str_in_ket = string[:N].upper()
str_op = string[0]+string[0].upper()
str_comb = string[1:N]+string[1:N].upper()
str_ein = str_in_bra+','+str_in_ket+','+str_op+'->'+str_comb
contraction = einsum(str_ein,psi_reshape,conj(psi_reshape),X)
for site in range(1,N):
    str_op = string[site]+string[site].upper()
    str_comb_prev = str_comb
    str_comb = string[site+1:N]+string[site+1:N].upper()
    str_ein = str_comb_prev+','+str_op+'->'+str_comb
    contraction = einsum(str_ein,contraction,X)
allSx = contraction

# Print Results -----------------------
print_str = '{}\t{}\t{}\t{}\t{}\t'.format(0,real(dot(psi,dot(H2,conj(psi)))),real(dot(psi,dot(H,conj(psi)))),EE,allSx)
for site in range(len(Sx)):
    print_str += '{}\t'.format(Sx[site])
for site in range(len(SxSx)):
    print_str += '{}\t'.format(SxSx[site])
print(print_str)

if time_evolve:
    ####################################################
    # Time Evolution
    psi_reshape = reshape(psi,(2**int(N/2),2**int(N/2)))
    (U,S,V) = svd(psi_reshape)
    EE = -dot(S**2,log2(S**2))
    for step in range(int(tf/dt)):
        # Do time step ------------------------
        psi = dot(eH2,psi)
        psi /= sqrt(dot(psi,conj(psi)))

        # Check Observables -------------------
        # Measure entropy
        psi_reshape = reshape(psi,(2**int(N/2),2**int(N/2)))
        (U,S,V) = svd(psi_reshape)
        EE = -dot(S**2,log2(S**2))
        # Measure <Sxi>
        psi_reshape = reshape(psi,(2,)*N)
        Sx = zeros(N)
        for site in range(N):
            string = 'abcdefghijklmnopqrstuvwxyz'
            str_in = string[:N]
            str_ot = string[:site]+string[site].upper()+string[site+1:N]
            str_op = string[site]+string[site].upper()
            str_ein= str_in+','+str_ot+','+str_op+'->'
            Sx[site] = einsum(str_ein,psi_reshape,conj(psi_reshape),X)
        # measure <Sx_i,Sx_{i+1}>
        SxSx = zeros(N-1)
        for site in range(N-1):
            string = 'abcdefghijklmnopqrstuvwxyz'
            str_in = string[:N]
            str_ot = string[:site]+string[site:site+2].upper()+string[site+2:N]
            str_op1= string[site]+string[site].upper()
            str_op2= string[site+1]+string[site+1].upper()
            str_ein= str_in+','+str_ot+','+str_op1+','+str_op2+'->'
            SxSx[site] = einsum(str_ein,psi_reshape,conj(psi_reshape),X,X)
        # measure <Sx1,Sx2,...,SxN>
        string = 'abcdefghijklmnopqrstuvwxyz'
        str_in_bra = string[:N]
        str_in_ket = string[:N].upper()
        str_op = string[0]+string[0].upper()
        str_comb = string[1:N]+string[1:N].upper()
        str_ein = str_in_bra+','+str_in_ket+','+str_op+'->'+str_comb
        contraction = einsum(str_ein,psi_reshape,conj(psi_reshape),X)
        for site in range(1,N):
            str_op = string[site]+string[site].upper()
            str_comb_prev = str_comb
            str_comb = string[site+1:N]+string[site+1:N].upper()
            str_ein = str_comb_prev+','+str_op+'->'+str_comb
            contraction = einsum(str_ein,contraction,X)
        allSx = contraction

        # Print Results -----------------------
        print_str = '{}\t{}\t{}\t{}\t{}\t'.format(dt*(step+1),real(dot(psi,dot(H2,conj(psi)))),real(dot(psi,dot(H,conj(psi)))),EE,allSx)
        for site in range(len(Sx)):
            print_str += '{}\t'.format(Sx[site])
        for site in range(len(SxSx)):
            print_str += '{}\t'.format(SxSx[site])
        print(print_str)
