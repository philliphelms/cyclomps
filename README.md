# cyclomps
An implementation of the density matrix renormalization group (DMRG)
algorithm using Cyclops Tensor Framework (CTF).

# Dependencies
Currently, the only dependency is the python version of 
CTF.

## To Do List
* Add noise to prevent convergence to local minima
* Implement 2-site algorithm
* Implement iDMRG (McCollough version)
* Add orthonormalization
* Fix end_gauge problem with continuing calculations
* Implement U(1) and SU(2) symmetries

## Known Bugs
* Must run test_algs.py and test_renorm.py separately
* There is an apparent memory leak.
* When using a large number of processors, there is some sort of memory corruption that occurs, setting all tensor elements to zero. 
