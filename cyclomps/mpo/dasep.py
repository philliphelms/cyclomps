import numpy as np

def get_ops(d):
    ops = dict()

    # Identity
    ops['I'] = np.eye(d)

    # Occupation operators
    for i in range(d):
        op = np.zeros((d, d))
        op[i,i] = 1
        ops['n'+str(i)] = op

    # Transfer operators
    for i in range(d):
        for j in range(d):
            if i != j:
                op = np.zeros((d, d))
                op[i, j] = 1
                ops['t'+str(i)+str(j)] = op

    # Return result
    return ops

def get_dasep_mpo(N, params, periodic=False):
    """
    Generate a quimb MatrixProductOperator representing
    the generator for the DASEP process.

    Args:
        N: int
            Number of sites
        params: dict
            The parameters for the generator as a dictionary,
            with entries:
                wA - Particle addition
                wD - Paritcle removal
                kA - Defect addition
                kD - Defect removal
                wmA - Particle addition on defect
                wmD - Particle removal on defect
                kmA - defect addition with particle on site
                kmD - defect removal with particle on site
                v - Forward hopping rate of particles
                vmp - Forward hopping rate into defect
                vmm - Forward hopping rate out of defect
                vm - Forward hopping rate between defects
    """
    # Get the needed operators
    ops = get_ops(4)

    # Generate the Bulk MPO
    Dmpo = 10
    d = 4
    ten = np.zeros((Dmpo, Dmpo, d, d), dtype=complex)
    ten[0, 0] = ops['I']
    ten[1, 0] = ops['t01']
    ten[2, 0] = ops['n0']
    ten[3, 0] = ops['t23']
    ten[4, 0] = ops['n2']
    ten[5, 0] = ops['t01']
    ten[6, 0] = ops['n0']
    ten[7, 0] = ops['t23']
    ten[8, 0] = ops['n2']
    ten[9, 0] = params['wA'] * ops['t01'] + \
               -params['wA'] * ops['n0'] + \
                params['wD'] * ops['t10'] + \
               -params['wD'] * ops['n1'] + \
                params['kA'] * ops['t02'] + \
               -params['kA'] * ops['n0'] + \
                params['kD'] * ops['t20'] + \
               -params['kD'] * ops['n2'] + \
                params['wmA'] * ops['t23'] + \
               -params['wmA'] * ops['n2'] + \
                params['wmD'] * ops['t32'] + \
               -params['wmD'] * ops['n3'] + \
                params['kmA'] * ops['t13'] + \
               -params['kmA'] * ops['n1'] + \
                params['kmD'] * ops['t31'] + \
               -params['kmD'] * ops['n3']

    ten[-1, 1] = params['v'] * ops['t10']
    ten[-1, 2] = - params['v'] * ops['n1']
    ten[-1, 3] = params['vmp'] * ops['t10']
    ten[-1, 4] = - params['vmp'] * ops['n1']
    ten[-1, 5] = params['vmm'] * ops['t32']
    ten[-1, 6] = - params['vmm'] * ops['n3']
    ten[-1, 7] = params['vm'] * ops['t32']
    ten[-1, 8] = - params['vm'] * ops['n3']
    ten[-1, 9] = ops['I']

    # Create a list to hold all MPO tensors
    mpo = [ten.copy() for _ in range(N)]
    mpo = [mpoi.transpose(0,2,3,1) for mpoi in mpo]
    print(mpo[0].shape)

    # Convert from a periodic MPO
    if periodic:
        raise ValueError('Periodic DMRG not implemented here')
    else:
        mpo[0] = mpo[0][-1,:]
        mpo[0] = np.array([mpo[0]])
        mpo[-1] = mpo[-1][:,:,:,0]
        mpo[-1] = np.array([mpo[-1]]).transpose(1,2,3,0)

    # Return result
    return [mpo]
