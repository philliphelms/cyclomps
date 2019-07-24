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
* Must run test_algs.py and test_renorm.py separately when using CTF
* Problems using eigh in state averaged renormalization - temporarily fixed by doing svd on the rdm
* CTF seems to have trouble using '-1' as an index - need to make sure we are avoiding this.
