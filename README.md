# cyclomps
An implementation of the density matrix renormalization group (DMRG)
algorithm using Cyclops Tensor Framework (CTF).

# Dependencies
Currently, the two main dependencies are the python version of 
CTF and, if using the davidson diagonalization routine, pyscf.

## To Do List
* Add noise to prevent convergence to local minima
* Implement 2-site algorithm
* Implement iDMRG (McCollough version)
* Add orthonormalization
* Check overlap after optimizations
* Fix end_gauge problem with continuing calculations
* Implement davidson using ctf/np
* Add more memory and time printing
* Implement U(1) and SU(2) symmetries
* Periodic BCs, (without simple cheat)
