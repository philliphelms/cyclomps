from numpy import ndarray as npndarray
from numpy import ones as npones
from numpy import zeros as npzeros
from numpy import float_
from cyclomps.mpo.ops import *
from cyclomps.tools.utils import *
from cyclomps.tools.mpo_tools import reorder_bonds
import collections
from numpy import exp

############################################################################
# General 2D Asymmetric Simple Exclusion Process:
#
#                        cd     du
#                         |     /\
#           ___ ___ ___ _\/ ___ _|_ ___ ___ ___ 
#          |   |   |   |   |   |   |   |   |   |
#          |___|___|___|___|___|___|___|___|___|
#          |   |   |   |   |   |   |   |   |   |
#          |___|___|___|___|___|___|<|ju___|___|
#          |   |   |  _jr_ |   |   |_| |   |   |
#          |___|___|_|_ _\/|___|___|___|___|___|
#          |   |   |   |   |   |   |   |   |   |
#          |___|___|___|___|___|___|___|___|___|
#    cr--->|   |   |   |   |   |/\___| |   |   |---> dr
#    dl<---|___|___|___|___|___|__jl___|___|___|<--- cl
#          |   |   |  _|   |   |   |   |   |   |
#          |___|___jd| |___|___|___|___|___|___|
#          |   |   | |>|   |   |   |   |   |   |
#          |___|___|___|___|___|___|___|___|___|
#          |   |   |   |   |   |   |   |   |   |
#          |___|___|___|___|___|___|___|___|___|
#                        |       /\
#                        \/      |
#                        dd      cu
#
# Right & Upwards directions are positive
#
# Functions:
#   return_mpo(N,hamParams):
#       hamParams[0] = jr  (Jump to the right)
#       hamParams[1] = jl  (Jump to the left)
#       hamParams[2] = ju  (Jump upwards)
#       hamParams[3] = jd  (Jump downwards)
#       hamParams[4] = cr  (Create via right hop)
#       hamParams[5] = cl  (Create via left hop)
#       hamParams[6] = cu  (Create via upwards hop)
#       hamParams[7] = cd  (Create via downwards hop)
#       hamParams[8] = dr  (Destroy via right hop)
#       hamParams[9] = dl  (Destroy via left hop)
#       hamParams[10]= du  (Destroy via upwards hop)
#       hamParams[11]= dd  (Destroy via downwards hop)
#       hamParams[12]= sx  (Bias in the x-direction)
#       hamParams[13]= sy  (Bias in the y-direction)
# 
# Note that for all of these parameters either a single value can be given
# for each, or a matrix can be given specifying the hopping rate at each
# site in the lattice instead of simply a general value. 
###########################################################################

##########################################################################
# Hamiltonian As MPO
##########################################################################

def return_mpo(N,hamParams,periodicx=False,periodicy=False):
    if hasattr(N,'__len__'):
        Nx = N[0]
        Ny = N[1]
    else:
        Nx = N
        Ny = N
    # Convert hamParams all to matrices
    if not isinstance(hamParams[0],(collections.Sequence,npndarray)):
        hamParams = val2matParams(Nx,Ny,hamParams)
    else:
        hamParams = extractParams(hamParams)
    # Generate MPO based on periodicity
    if periodicx and periodicy:
        mpo = periodic_xy_mpo(Nx,Ny,hamParams)
    elif periodicx:
        mpo = periodic_x_mpo(Nx,Ny,hamParams)
    elif periodicy:
        mpo = periodic_y_mpo(Nx,Ny,hamParams)
    else:
        mpo = open_mpo(Nx,Ny,hamParams)
    mpo = reorder_bonds(mpo)
    return mpo

def open_mpo(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 10+(Ny-2)*4
    mpiprint(5,'Hamiltonian Bond Dimension = {}'.format(ham_dim))
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = ejr[xi-1,yi]*Sm
            gen_mpo[Ny,0,:,:] = ejd[xi,yi-1]*Sm
            gen_mpo[Ny+1,0,:,:] = jr[xi-1,yi]*v
            gen_mpo[2*Ny,0,:,:] = jd[xi,yi-1]*v
            gen_mpo[2*Ny+1,0,:,:] = ejl[xi,yi]*Sp
            gen_mpo[3*Ny,0,:,:] = eju[xi,yi]*Sp
            gen_mpo[3*Ny+1,0,:,:] = jl[xi,yi]*n
            gen_mpo[4*Ny,0,:,:] = ju[xi,yi]*n
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(4): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = -n
            gen_mpo[ham_dim-1,3*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,4*Ny,:,:] = -v
            gen_mpo[ham_dim-1,4*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecr[xi,yi] + ecl[xi,yi] + ecd[xi,yi] + ecu[xi,yi])*Sm -\
                                 ( cr[xi,yi] +  cl[xi,yi] +  cd[xi,yi] +  cu[xi,yi])*v  +\
                                 (edr[xi,yi] + edl[xi,yi] + edd[xi,yi] + edu[xi,yi])*Sp -\
                                 ( dr[xi,yi] +  dl[xi,yi] +  dd[xi,yi] +  du[xi,yi])*n
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
                gen_mpo[3*Ny,0,:,:] = z
                gen_mpo[4*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def periodic_xy_mpo(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_mpo(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[jr[xind2,yind2]*n]])
                op2[inds[0]] = array([[-v]])
                mpoL.append(op1)
                mpoL.append(op2)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[jl[xind1,yind1]*v]])
                op2[inds[0]] = array([[-n]])
                mpoL.append(op1)
                mpoL.append(op2)
        else: # Vertical
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[jd[xind2,yind2]*v]])
                op2[inds[0]] = array([[-n]])
                mpoL.append(op1)
                mpoL.append(op2)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[ju[xind1,yind1]*n]])
                op2[inds[0]] = array([[-v]])
                mpoL.append(op1)
                mpoL.append(op2)
    return mpoL

def periodic_x_mpo(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_mpo(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[jr[xind2,yind2]*n]])
                op2[inds[0]] = array([[-v]])
                mpoL.append(op1)
                mpoL.append(op2)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[jl[xind1,yind1]*v]])
                op2[inds[0]] = array([[-n]])
                mpoL.append(op1)
                mpoL.append(op2)
    return mpoL

def periodic_y_mpo(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_mpo(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'vert':
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[jd[xind2,yind2]*v]])
                op2[inds[0]] = array([[-n]])
                mpoL.append(op1)
                mpoL.append(op2)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                op2 = [None]*(Nx*Ny)
                op2[inds[1]] = array([[ju[xind1,yind1]*n]])
                op2[inds[0]] = array([[-v]])
                mpoL.append(op1)
                mpoL.append(op2)
    return mpoL

##########################################################################
# Useful Functions
##########################################################################

def exponentiateBias(hamParams):
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    ejr = jr*exp(sx)
    ejl = jl*exp(-sx)
    eju = ju*exp(sy)
    ejd = jd*exp(-sy)
    ecr = cr*exp(sx)
    ecl = cl*exp(-sx)
    ecu = cu*exp(sy)
    ecd = cd*exp(-sy)
    edr = dr*exp(sx)
    edl = dl*exp(-sx)
    edu = du*exp(sy)
    edd = dd*exp(-sy)
    return (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd)

def val2matParams(Nx,Ny,hamParams):
    jr = hamParams[0]
    jl = hamParams[1]
    ju = hamParams[2]
    jd = hamParams[3]
    cr = hamParams[4]
    cl = hamParams[5]
    cu = hamParams[6]
    cd = hamParams[7]
    dr = hamParams[8]
    dl = hamParams[9]
    du = hamParams[10]
    dd = hamParams[11]
    sx = hamParams[12]
    sy = hamParams[13]
    # Set interior hopping rates
    jr_m = jr*npones((Nx,Ny),dtype=float_)
    jl_m = jl*npones((Nx,Ny),dtype=float_)
    ju_m = ju*npones((Nx,Ny),dtype=float_)
    jd_m = jd*npones((Nx,Ny),dtype=float_)
    # Initialize Matrices for insertion & removal rates
    cr_m = npzeros((Nx,Ny),dtype=float_)
    cl_m = npzeros((Nx,Ny),dtype=float_)
    cu_m = npzeros((Nx,Ny),dtype=float_)
    cd_m = npzeros((Nx,Ny),dtype=float_)
    dr_m = npzeros((Nx,Ny),dtype=float_)
    dl_m = npzeros((Nx,Ny),dtype=float_)
    du_m = npzeros((Nx,Ny),dtype=float_)
    dd_m = npzeros((Nx,Ny),dtype=float_)
    # Set appropriate boundary terms
    cr_m[0,:] = cr
    cl_m[-1,:] = cl
    cu_m[:,-1] = cu
    cd_m[:,0] = cd
    dr_m[-1,:] = dr
    dl_m[0,:] = dl
    du_m[:,0] = du
    dd_m[:,-1] = dd
    # Set bias
    sx_m = sx*npones((Nx,Ny),dtype=float_)
    sy_m = sy*npones((Nx,Ny),dtype=float_)
    return (jr_m,jl_m,ju_m,jd_m,cr_m,cl_m,cu_m,cd_m,dr_m,dl_m,du_m,dd_m,sx_m,sy_m)

def extractParams(hamParams):
    jr = hamParams[0].astype(dtype=float_)
    jl = hamParams[1].astype(dtype=float_)
    ju = hamParams[2].astype(dtype=float_)
    jd = hamParams[3].astype(dtype=float_)
    cr = hamParams[4].astype(dtype=float_)
    cl = hamParams[5].astype(dtype=float_)
    cu = hamParams[6].astype(dtype=float_)
    cd = hamParams[7].astype(dtype=float_)
    dr = hamParams[8].astype(dtype=float_)
    dl = hamParams[9].astype(dtype=float_)
    du = hamParams[10].astype(dtype=float_)
    dd = hamParams[11].astype(dtype=float_)
    sx = hamParams[12].astype(dtype=float_)
    sy = hamParams[13].astype(dtype=float_)
    return (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)

##########################################################################
# Current MPOS
##########################################################################

def curr_mpo(N,hamParams,
             periodicx=False,periodicy=False,
             includex=True,includey=True,
             singleBond=False,xbond=None,ybond=None,orientation=None):
    if hasattr(N,'__len__'):
        Nx = N[0]
        Ny = N[1]
    else:
        Nx = N
        Ny = N
    # Convert hamParams all to matrices
    if not isinstance(hamParams[0],(collections.Sequence,npndarray)):
        hamParams = val2matParams(Nx,Ny,hamParams)
    else:
        hamParams = extractParams(hamParams)
    # Generate MPO based on periodicity
    if singleBond:
        mpo = single_bond_curr(Nx,Ny,hamParams,xbond,ybond,orientation)
    else:
        if includex and includey:
            if periodicx and periodicy:
                mpo = periodic_xy_curr_xy(Nx,Ny,hamParams)
            elif periodicx:
                mpo = periodic_x_curr_xy(Nx,Ny,hamParams)
            elif periodicy:
                mpo = periodic_y_curr_xy(Nx,Ny,hamParams)
            else:
                mpo = open_curr_xy(Nx,Ny,hamParams)
        elif includex:
            if periodicx and periodicy:
                mpo = periodic_xy_curr_x(Nx,Ny,hamParams)
            elif periodicx:
                mpo = periodic_x_curr_x(Nx,Ny,hamParams)
            elif periodicy:
                mpo = periodic_y_curr_x(Nx,Ny,hamParams)
            else:
                mpo = open_curr_x(Nx,Ny,hamParams)
        elif includey:
            if periodicx and periodicy:
                mpo = periodic_xy_curr_y(Nx,Ny,hamParams)
            elif periodicx:
                mpo = periodic_x_curr_y(Nx,Ny,hamParams)
            elif periodicy:
                mpo = periodic_y_curr_y(Nx,Ny,hamParams)
            else:
                mpo = open_curr_y(Nx,Ny,hamParams)
    return mpo

def single_bond_curr(Nx,Ny,hamParams,xbond,ybond,orientation):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    # Set all params to zero, except those involved
    if orientation == 'vert':
        jr = npzeros(jr.shape)
        jl = npzeros(jl.shape)
        cr = npzeros(jl.shape)
        cl = npzeros(jl.shape)
        dr = npzeros(jl.shape)
        dl = npzeros(jl.shape)
        if ybond == 'top':
            ju = npzeros(ju.shape)
            jd = npzeros(jd.shape)
            cu = npzeros(cu.shape)
            dd = npzeros(dd.shape)
            mask = npones(cd.shape,bool)
            mask[xbond,0] = False
            cd[mask] = 0.
            du[mask] = 0. 
        elif ybond == 'bottom':
            ju = npzeros(ju.shape)
            jd = npzeros(jd.shape)
            cd = npzeros(cd.shape)
            du = npzeros(du.shape)
            mask = npones(cu.shape,bool)
            mask[xbond,-1] = False
            cu[mask] = 0.
            dd[mask] = 0.
        else:
            cu = npzeros(cu.shape)
            cd = npzeros(cd.shape)
            du = npzeros(du.shape)
            dd = npzeros(dd.shape)
            mask = npones(ju.shape,bool)
            mask[xbond,ybond+1] = False
            ju[mask] = 0.
            mask = npones(jd.shape,bool)
            mask[xbond,ybond] = False
            jd[mask] = 0.
    elif orientation == 'horz':
        ju = npzeros(jr.shape)
        jd = npzeros(jl.shape)
        cu = npzeros(jl.shape)
        cd = npzeros(jl.shape)
        du = npzeros(jl.shape)
        dd = npzeros(jl.shape)
        if xbond == 'left':
            jr = npzeros(jr.shape)
            jl = npzeros(jl.shape)
            cl = npzeros(cl.shape)
            dr = npzeros(dr.shape)
            mask = npones(dl.shape,bool)
            mask[0,ybond] = False
            dl[mask] = 0.
            cr[mask] = 0.
        elif xbond == 'right':
            jr = npzeros(jr.shape)
            jl = npzeros(jl.shape)
            dl = npzeros(dl.shape)
            cr = npzeros(cr.shape)
            mask = npones(dr.shape,bool)
            mask[-1,ybond] = False
            dr[mask] = 0.
            cl[mask] = 0.
        else:
            cr = npzeros(cu.shape)
            cl = npzeros(cd.shape)
            dr = npzeros(du.shape)
            dl = npzeros(dd.shape)
            mask = npones(jr.shape,bool)
            mask[xbond,ybond] = False
            jr[mask] = 0.
            mask = npones(jl.shape,bool)
            mask[xbond+1,ybond] = False
            jl[mask] = 0.
    hamParams = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2+2*Ny
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = ejr[xi-1,yi]*Sm
            gen_mpo[Ny,0,:,:] = -ejd[xi,yi-1]*Sm
            gen_mpo[Ny+1,0,:,:] = -ejl[xi,yi]*Sp
            gen_mpo[2*Ny,0,:,:] = eju[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecr[xi,yi] - ecl[xi,yi] - ecd[xi,yi] + ecu[xi,yi])*Sm +\
                                 (edr[xi,yi] - edl[xi,yi] - edd[xi,yi] + edu[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def open_curr_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2+2*Ny
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = ejr[xi-1,yi]*Sm
            gen_mpo[Ny,0,:,:] = -ejd[xi,yi-1]*Sm
            gen_mpo[Ny+1,0,:,:] = -ejl[xi,yi]*Sp
            gen_mpo[2*Ny,0,:,:] = eju[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecr[xi,yi] - ecl[xi,yi] - ecd[xi,yi] + ecu[xi,yi])*Sm +\
                                 (edr[xi,yi] - edl[xi,yi] - edd[xi,yi] + edu[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def open_curr_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2*Ny+2
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = ejr[xi-1,yi]*Sm
            gen_mpo[Ny+1,0,:,:] = -ejl[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecr[xi,yi] - ecl[xi,yi])*Sm +\
                                 (edr[xi,yi] - edl[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def open_curr_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2*Ny+2
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[Ny,0,:,:] = -ejd[xi,yi-1]*Sm
            gen_mpo[2*Ny,0,:,:] = eju[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (-ecd[xi,yi] + ecu[xi,yi])*Sm +\
                                 (-edd[xi,yi] + edu[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def periodic_xy_curr_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_xy(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
        else: # Vertical
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL

def periodic_xy_curr_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_x(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
    return mpoL

def periodic_xy_curr_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_y(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            pass
        else: # Vertical
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL

def periodic_x_curr_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_xy(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
    return mpoL

def periodic_x_curr_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_x(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
    return mpoL

def periodic_x_curr_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_y(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            pass
    return mpoL

def periodic_y_curr_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_xy(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'vert':
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL

def periodic_y_curr_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_x(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'vert':
            pass
    return mpoL

def periodic_y_curr_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_curr_y(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'vert':
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[-ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL

##########################################################################
# Activity MPOS
##########################################################################

def act_mpo(N,hamParams,
            periodicx=False,periodicy=False,
            includex=True,includey=True,
            singleBond=False,xbond=None,ybond=None,orientation=None):
    if hasattr(N,'__len__'):
        Nx = N[0]
        Ny = N[1]
    else:
        Nx = N
        Ny = N
    # Convert hamParams all to matrices
    if not isinstance(hamParams[0],(collections.Sequence,npndarray)):
        hamParams = val2matParams(Nx,Ny,hamParams)
    else:
        hamParams = extractParams(hamParams)
    # Generate MPO based on periodicity
    if singleBond:
        mpo = single_bond_act(Nx,Ny,hamParams,xbond,ybond,orientation)
    else:
        if includex and includey:
            if periodicx and periodicy:
                mpo = periodic_xy_act_xy(Nx,Ny,hamParams)
            elif periodicx:
                mpo = periodic_x_act_xy(Nx,Ny,hamParams)
            elif periodicy:
                mpo = periodic_y_act_xy(Nx,Ny,hamParams)
            else:
                mpo = open_act_xy(Nx,Ny,hamParams)
        elif includex:
            if periodicx and periodicy:
                mpo = periodic_xy_act_x(Nx,Ny,hamParams)
            elif periodicx:
                mpo = periodic_x_act_x(Nx,Ny,hamParams)
            elif periodicy:
                mpo = periodic_y_act_x(Nx,Ny,hamParams)
            else:
                mpo = open_act_x(Nx,Ny,hamParams)
        elif includey:
            if periodicx and periodicy:
                mpo = periodic_xy_act_y(Nx,Ny,hamParams)
            elif periodicx:
                mpo = periodic_x_act_y(Nx,Ny,hamParams)
            elif periodicy:
                mpo = periodic_y_act_y(Nx,Ny,hamParams)
            else:
                mpo = open_act_y(Nx,Ny,hamParams)
    return mpo

def single_bond_act(Nx,Ny,hamParams,xbond,ybond,orientation):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    # Set all params to zero, except those involved
    if orientation == 'vert':
        jr = npzeros(jr.shape)
        jl = npzeros(jl.shape)
        cr = npzeros(jl.shape)
        cl = npzeros(jl.shape)
        dr = npzeros(jl.shape)
        dl = npzeros(jl.shape)
        if ybond == 'top':
            ju = npzeros(ju.shape)
            jd = npzeros(jd.shape)
            cu = npzeros(cu.shape)
            dd = npzeros(dd.shape)
            mask = npones(cd.shape,bool)
            mask[xbond,0] = False
            cd[mask] = 0.
            du[mask] = 0. 
        elif ybond == 'bottom':
            ju = npzeros(ju.shape)
            jd = npzeros(jd.shape)
            cd = npzeros(cd.shape)
            du = npzeros(du.shape)
            mask = npones(cu.shape,bool)
            mask[xbond,-1] = False
            cu[mask] = 0.
            dd[mask] = 0.
        else:
            cu = npzeros(cu.shape)
            cd = npzeros(cd.shape)
            du = npzeros(du.shape)
            dd = npzeros(dd.shape)
            mask = npones(ju.shape,bool)
            mask[xbond,ybond+1] = False
            ju[mask] = 0.
            mask = npones(jd.shape,bool)
            mask[xbond,ybond] = False
            jd[mask] = 0.
    elif orientation == 'horz':
        ju = npzeros(jr.shape)
        jd = npzeros(jl.shape)
        cu = npzeros(jl.shape)
        cd = npzeros(jl.shape)
        du = npzeros(jl.shape)
        dd = npzeros(jl.shape)
        if xbond == 'left':
            jr = npzeros(jr.shape)
            jl = npzeros(jl.shape)
            cl = npzeros(cl.shape)
            dr = npzeros(dr.shape)
            mask = npones(dl.shape,bool)
            mask[0,ybond] = False
            dl[mask] = 0.
            cr[mask] = 0.
        elif xbond == 'right':
            jr = npzeros(jr.shape)
            jl = npzeros(jl.shape)
            dl = npzeros(dl.shape)
            cr = npzeros(cr.shape)
            mask = npones(dr.shape,bool)
            mask[-1,ybond] = False
            dr[mask] = 0.
            cl[mask] = 0.
        else:
            cr = npzeros(cu.shape)
            cl = npzeros(cd.shape)
            dr = npzeros(du.shape)
            dl = npzeros(dd.shape)
            mask = npones(jr.shape,bool)
            mask[xbond,ybond] = False
            jr[mask] = 0.
            mask = npones(jl.shape,bool)
            mask[xbond+1,ybond] = False
            jl[mask] = 0.
    hamParams = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2+2*Ny
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = ejr[xi-1,yi]*Sm
            gen_mpo[Ny,0,:,:] = ejd[xi,yi-1]*Sm
            gen_mpo[Ny+1,0,:,:] = ejl[xi,yi]*Sp
            gen_mpo[2*Ny,0,:,:] = eju[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecr[xi,yi] + ecl[xi,yi] + ecd[xi,yi] + ecu[xi,yi])*Sm +\
                                 (edr[xi,yi] + edl[xi,yi] + edd[xi,yi] + edu[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def open_act_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2+2*Ny
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = ejr[xi-1,yi]*Sm
            gen_mpo[Ny,0,:,:] = ejd[xi,yi-1]*Sm
            gen_mpo[Ny+1,0,:,:] = ejl[xi,yi]*Sp
            gen_mpo[2*Ny,0,:,:] = eju[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecr[xi,yi] + ecl[xi,yi] + ecd[xi,yi] + ecu[xi,yi])*Sm +\
                                 (edr[xi,yi] + edl[xi,yi] + edd[xi,yi] + edu[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def open_act_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2*Ny+2
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[1,0,:,:] = ejr[xi-1,yi]*Sm
            gen_mpo[Ny+1,0,:,:] = ejl[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecr[xi,yi] + ecl[xi,yi])*Sm +\
                                 (edr[xi,yi] + edl[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def open_act_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # List to hold all MPOs
    mpoL = []
    # Main MPO
    mpo = []
    ham_dim = 2*Ny+2
    for xi in range(Nx):
        for yi in range(Ny):
            # Build generic MPO
            gen_mpo = zeros((ham_dim,ham_dim,2,2))
            gen_mpo[0,0,:,:] = I 
            gen_mpo[Ny,0,:,:] = ejd[xi,yi-1]*Sm
            gen_mpo[2*Ny,0,:,:] = eju[xi,yi]*Sp
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for k in range(2): 
                for l in range(Ny-1):
                    gen_mpo[row_ind,col_ind,:,:] = I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            gen_mpo[ham_dim-1,Ny,:,:] = Sp
            gen_mpo[ham_dim-1,2*Ny,:,:] = Sm
            gen_mpo[ham_dim-1,2*Ny+1,:,:] = I
            # Include creation & annihilation
            gen_mpo[ham_dim-1,0,:,:] += (ecd[xi,yi] + ecu[xi,yi])*Sm +\
                                 (edd[xi,yi] + edu[xi,yi])*Sp
            # Prevent interaction between ends
            if (yi == 0) and (xi != 0):
                gen_mpo[Ny,0,:,:] = z
                gen_mpo[2*Ny,0,:,:] = z
            # Add operator to list of operators
            if (xi == 0) and (yi == 0):
                mpo.append(expand_dims(gen_mpo[ham_dim-1,:],0))
            elif (xi == Nx-1) and (yi == Ny-1):
                mpo.append(expand_dims(gen_mpo[:,0],1))
            else:
                mpo.append(gen_mpo)
    mpoL.append(mpo)
    return mpoL

def periodic_xy_act_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_xy(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
        else: # Vertical
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL

def periodic_xy_act_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_x(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
    return mpoL

def periodic_xy_act_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_y(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            pass
        else: # Vertical
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL

def periodic_x_act_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_xy(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
    return mpoL

def periodic_x_act_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_x(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            yind1 = inds[0]
            xind1 = 0
            yind2 = inds[0]
            xind2 = -1
            if jr[xind2,yind2] != 0:
                # Jump right
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejr[xind2,yind2]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
            if jl[xind1,yind1] != 0:
                # Jump left
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejl[xind1,yind1]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
    return mpoL

def periodic_x_act_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_y(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along x-axis
    for yi in range(Ny):
        coupled_sites.append([yi,Ny*(Nx-1)+yi,'horz'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'horz':
            pass
    return mpoL

def periodic_y_act_xy(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_xy(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'vert':
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL

def periodic_y_act_x(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_x(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'vert':
            pass
    return mpoL

def periodic_y_act_y(Nx,Ny,hamParams):
    # Extract parameter Values
    (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy) = hamParams
    (ejr,ejl,eju,ejd,ecr,ecl,ecu,ecd,edr,edl,edu,edd) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_act_y(Nx,Ny,hamParams)
    # Container to hold coupled sites
    coupled_sites = []
    # Periodic coupling along y-axis
    for xi in range(Nx):
        coupled_sites.append([Ny*(xi+1)-1,Ny*xi,'vert'])
    # Build all related operator
    for i in range(len(coupled_sites)):
        inds = coupled_sites[i][:2]
        if coupled_sites[i][2] == 'vert':
            xind1 = int(inds[1]/Ny)
            yind1 = 0
            xind2 = int(inds[1]/Ny)
            yind2 = -1
            if jd[xind2,yind2] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[ejd[xind2,yind2]*Sm]])
                op1[inds[0]] = array([[Sp]])
                mpoL.append(op1)
            if ju[xind1,yind1] != 0:
                op1 = [None]*(Nx*Ny)
                op1[inds[1]] = array([[eju[xind1,yind1]*Sp]])
                op1[inds[0]] = array([[Sm]])
                mpoL.append(op1)
    return mpoL
