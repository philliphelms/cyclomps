from cyclomps.tools.utils import *
from cyclomps.algs.ed import ed
from cyclomps.mpo.asep2D import return_mpo,curr_mpo
from numpy import complex as complex_
from sys import argv
from cyclomps.mpo.ops import *
from cyclomps.tools.mpo_tools import mpo2mat, reorder_bonds

# System Size
Ny = 3
Nx = 3

# Hamiltonian Parameters
#jr = 0.9
#jl = 1.-jr
#ju = 0.9
#jd = 1.-ju
#cr = 0.5
#cl = 0.5
#cu = 0.5
#cd = 0.5
#dr = 0.5
#dl = 0.5
#du = 0.5
#dd = 0.5
#sx = 0.5
#sy = 0.
jr = 0.9
jl = 1.-jr
ju = 0.
jd = 0.
cr = 0.5
cl = 0.5
cu = 0.
cd = 0.
dr = 0.5
dl = 0.5
du = 0.
dd = 0.
sx = 0.5
sy = 0.

# Set up MPO -------------------------------
periodicy = False
periodicx = False
hamParams = array([jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy])
mpo = return_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False)

# Run diagonalization -----------------------
H = mpo2mat(mpo)
E0,vl,vr = ed(mpo,left=True)
vr0 = vr[:,0]
vl0 = vl[:,0]
print('E0 = {}'.format(E0[0].real))

# Evaluate random parameters ---------------
# Local Density
density = zeros((Nx,Ny))
for x in range(Nx):
    for y in range(Ny):
        rho_mpo = [array([[I]])]*(Nx*Ny)
        rho_mpo[x*Nx+y] = array([[n]])
        rho_mpo = [rho_mpo]
        rho_mpo = reorder_bonds(rho_mpo)
        rho = mpo2mat(rho_mpo)
        density[x,y] = dot(vl0.conj(),dot(rho,vr0)).real
        print('Density({},{}) = {}'.format(x,y,density[x,y]))
print('Total Density = {}'.format(summ(density)))
# Local Vertical Current
ycurr = 0.
for x in range(Nx):
    for y in ['bottom'] + range(Ny-1) + ['top']:
        cmpo = curr_mpo((Nx,Ny),hamParams,periodicx=False,periodicy=False,singleBond=True,orientation='vert',xbond=x,ybond=y)
        curr = mpo2mat(cmpo)
        curr = dot(vl0.conj(),dot(curr,vr0)).real
        print('Vertical Current({},{}) = {}'.format(x,y,curr))
        ycurr += curr
print('Total Vertical Current = {}'.format(ycurr))
# Local Horizontal Current
xcurr = 0.
for x in ['left'] + range(Nx-1) + ['right']:
    for y in range(Ny):
        cmpo = curr_mpo((Nx,Ny),hamParams,periodicx=False,periodicy=False,singleBond=True,orientation='horz',xbond=x,ybond=y)
        curr = mpo2mat(cmpo)
        curr = dot(vl0.conj(),dot(curr,vr0)).real
        print('Horizontal Current({},{}) = {}'.format(x,y,curr))
        xcurr += curr
print('Total Horizontal Current = {}'.format(xcurr))

