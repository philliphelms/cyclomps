"""
Tools for linear algebra

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

.. To Do:

"""

from cyclomps.tools.utils import *
from cyclomps.tools.params import *
import tempfile
import h5py
from numpy import argmax as npargmax
from numpy import ndarray as npndarray
import numpy as np

def pick_real_eigs(w, v, nroots, x0):
    # Pick the eigenvalue with the smallest imaginary component
    # where we are forced to choose at least one eigenvalue.
    abs_imag = abss(imag(w))
    max_imag_tol = max((abs_imag < max_imag_tol))[0]
    if len(realidx) < nroots:
        idx = w.real.argsort() # PH - Problem with ctf, no argsort
    else:
        idx = realidx[w[realidx].real.argsort()]
        return w[idx].real, v[:,idx].real, idx

def davidson(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
        lindep=1e-14, callback=None, nroots=1, lessio=False, left=False,
        pick=pick_real_eigs, follow_state=False):
    """
    Davidson diagonalization for non-hermitian eigenproblem. 

    Adapted from pyscf's version, implemented by Qiming Sun <osirpt.sun@gmail.com>
    """
    res = davidson_nosym1(lambda xs: [aop(x) for x in xs],
                          x0, precond, tol, max_cycle, max_space, lindep,
                          callback, nroots, lessio, left, 
                          pick, follow_state)

    if left:
        e, vl, vr = res[1:]
        if nroots == 1:
            return e[0], vl[0], vr[0]
        else:
            return e, vl, vr
    else:
        e, x = res[1:]
        if nroots == 1:
            return e[0], x[0]
        else:
            return e, x

def davidson_nosym1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
                    lindep=1e-14, callback=None, nroots=1, lessio=False, left=False,
                    pick=pick_real_eigs, follow_state=False):

    # Format input guess correctly
    if (isinstance(x0,npndarray) or isinstance(x0,ctf.tensor)) and (x0.ndim == 1):
        x0 = [x0]

    # Initial Parameters
    heff = None
    fresh_start = True
    e = 0
    v = None
    conv = [False] * nroots
    emin = None
    max_space = max_space + nroots * 3
    toloose = np.sqrt(tol)

    # Start diagonalization procedure
    for icyc in range(max_cycle):
        if fresh_start:
            xs = []#_Xlist()
            ax = []#_Xlist()
            space = 0
            xt = None
            xt, x0 = _qr(x0), None
            max_dx_last = 1e9 # PH - Might add to func args
        elif (len(xt) > 1):
            xt = _qr(xt)
            xt = xt[:40] # 40 trial vectors at most # PH - Might add to function args?

        # Multiply hamiltonian vs vector (Hx) and save results
        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space+rnow

        # Initialize effective hamiltonian (heff)
        if heff is None:
            heff = zeros((max_space+nroots,max_space+nroots), dtype=axt[0].dtype)
        else:
            pass
            #heff = array(heff,dtype=axt[0].dtype)

        # Save previous results
        elast = e
        vlast = v
        conv_last = conv

        # Populate effective hamiltonian
        for i in range(rnow):
            for k in range(rnow):
                heff[head+k,head+i] = to_nparray(dot(conj(xt[k]), axt[i])) # PH - to_nparray because of ctf issues
        for i in range(head):
            axi = ax[i]
            xi = xs[i]
            for k in range(rnow):
                heff[head+k,i] = to_nparray(dot(conj(xt[k]), axi))
                heff[i,head+k] = to_nparray(dot(conj(xi), axt[k]))

        # Solve eigenproblem
        w, v = eig(heff[:space,:space])
        e,v,idx = pick(w,v,nroots,locals())
        e = e[:nroots]
        v = v[:,:nroots]

        # Save results?
        x0 = _gen_x0(v, xs)
        ax0 = _gen_x0(v, ax)

        # Sort
        elast, conv_last = _sort_elast(elast, conv_last, vlast, v, fresh_start)
        try:
            de = e-elast[:len(e)]
        except:
            de = e-elast
        dx_norm = []
        xt = []

        for k in range(e.shape[0]):
            ek = e[k]
            xt.append(ax0[k] - ek * x0[k])
            norm = sqrt(real(dot(conj(xt[k]), xt[k])))
            norm = to_nparray( take(norm,0) )
            dx_norm.append((norm))
        conv = (dx_norm < toloose) & (de < tol)

        # Check for convergence and restart if needed
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = npargmax(abss(de))
        if all(conv):
            break
        elif (follow_state and max_dx_norm > 1 and max_dx_norm/max_dx_last > 3 and space>nroots*3):
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        # Remove subspace linear dependency
        if any(((not conv[k]) and n**2>lindep) for k, n in enumerate(dx_norm)):
            for k, ek in enumerate(e):
                if (not conv[k]) and dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1./sqrt(real(dot(conj(xt[k]), xt[k])))
                else:
                    xt[k] = None
        else:
            for k, ek in enumerate(e):
                if dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1./sqrt(dot(conj(xt[k]), xt[k]).real)
                else:
                    xt[k] = None
        # Remove those that don't pass muster
        xt = [xi for xi in xt if xi is not None]
        for i in range(space):
            xsi = xs[i]
            for xi in xt:
                xi -= xsi * dot(conj(xsi), xi)
        norm_min = 1.
        for i,xi in enumerate(xt):
            norm = to_nparray(sqrt(real(dot(conj(xi),xi))))
            if norm**2. > lindep:
                xt[i] *= 1./norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        # Remove those that don't pass muster
        xt = [xi for xi in xt if xi is not None]
        xi = None
        # Warning if none remain
        if len(xt) == 0:
            mpiprint(5,'Linear dependency in trial subspace')
            conv = [conv[k] or (norm < toloose) for k,norm in enumerate(dx_norm)]
            break

        # Save previous results
        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        # Call callback function
        if callable(callback):
            callback(locals())

    # Return results (and calc left if needed)
    if left:
        w,vl,v = eig(heff[:space,:space], left=True)
        e,v,idx = pick(w,v,nroots,x0)
        xl = _gen_x0(conj(vl[:,idx[:nroots]]),xs)
        x0 = _gen_x0(v[:,:nroots], xs)
        return conv, e[:nroots], xl, x0
    else:
        return conv, e, x0

def _sort_elast(elast, conv_last, vlast, v, fresh_start):
    if fresh_start:
        return elast, conv_last
    head, nroots = vlast.shape
    ovlp = abss(dot(transpose(conj(v[:head])), vlast))
    ovlp = to_nparray(ovlp)
    idx = npargmax(ovlp, axis=1)
    return [elast[i] for i in idx], [conv_last[i] for i in idx]

def _qr(xs):
    norm = sqrt(real(dot(conj(xs[0]), xs[0])))
    qs = [xs[0]/norm]
    for i in range(1, len(xs)):
        xi = xs[i].copy() + 0.j
        for j in range(len(qs)):
            xi -= (qs[j]+0.j) * dot(conj((qs[j]+0.j)), xi)
        norm = to_nparray(sqrt(real(dot(conj(xi),xi))))
        if norm > 1e-7:
            qs.append(xi/norm)
    return qs

def _gen_x0(v, xs):
    space, nroots = v.shape
    x0 = []
    for k in range(nroots):
        x0.append(xs[space-1] * v[space-1,k])
    for i in reversed(range(space-1)):
        xsi = xs[i]
        for k in range(nroots):
            x0[k] += v[i,k] * xsi
    return x0

# Additional Class for out of core storage
class _Xlist(list):
    def __init__(self):
        self.scr_h5 = H5TmpFile()
        self.index = []

    def __getitem__(self, n):
        key = self.index[n]
        if USE_CTF:
            return from_nparray(self.scr_h5[key].value)
        else:
            return self.scr_h5[key].value
    
    def append(self, x):
        key = str(len(self.index) + 1)
        if key in self.index:
            for i in range(len(self.index)+1):
                if str(i) not in self.index:
                    key = str(i)
                    break
        self.index.append(key)
        if USE_CTF:
            self.scr_h5[key] = to_nparray(x)
        else:
            self.scr_h5[key] = x
        self.scr_h5.flush()

    def __setitem__(self, n, x):
        key = self.index[n]
        # PH - more correct way of doing this
        if USE_CTF:
            self.scr_h5[key][:] = to_nparray(x)
        else:
            self.scr_h5[key][:] = x
        self.scr_h5.flush()

    def __len__(self):
        return len(self.index)

    def pop(self, index):
        key = self.index.pop(index)
        del(self.scr_h5[key])

class H5TmpFile(h5py.File):
    def __init__(self, filename=None, *args, **kwargs):
        if filename is None:
            tmpfile = tempfile.NamedTemporaryFile(dir=CALCDIR)
            filename = tmpfile.name
        h5py.File.__init__(self, filename, *args, **kwargs)

    def __del__(self):
        self.close()
