# cyclomps
Matrix Product States (MPS) and Density Matrix Renormalization Group (DMRG) 
implementation using Cyclops Tensor Framework (CTF).

## Python Operations for Cyclops Tensors
Below is a brief summary of the available CTF functions in the python 
implementation.
### Creating CTF Objects
#### `ctf.astensor()`
Acts like `np.array` but creating ctf tensors.
#### `ctf.from_nparray()` or `ctf.array()`
Converts a numpy array into a ctf tensor.
#### `ctf.copy()`
Returns of copy of input tensor. 
Not a pointer/reference to original tensor.
#### `ctf.eye()` or `ctf.identity()`
Returns a 2D identity tensor.
#### `ctf.ones`
Create a tensor filled with ones. 
#### `ctf.to_nparray()`
Convert a ctf tensor to a numpy array
#### `ctf.zeros()`
Create a tensor filled with zeros.
#### `ctf.random.random()`
Create a tensor filled with pseudo-randomly generated numbers.

### Elementwise operations on CTF Objects
#### `ctf.abs()`
Elementwise absolute value.
#### `ctf.conj()`
Elementwise complex conjugate of input tensor.
#### `ctf.exp()`
Elementwise exponential of input tensor.
#### `ctf.imag()`
Elementwise Imaginary part of tensor.
#### `ctf.real()`
Elementwise Real part of tensor.
#### `ctf.power(A,B)`
Elementwise raising tensor A to powers in tensor B.
 
# Reshaping-type CTF operations
#### `ctf.diag()`
Returns diagonal elements as 1D tensor if given 2D input 
or puts 1D input onto diagonal of returned 2D tensor.
#### `ctf.diagonal()`
Allows getting diagonal of >2D tensor, if square and axes are specified.
#### `ctf.hstack`
Stack tensors column-wise.
#### `ctf.vstack`
Stack tensors row-wise.
#### `ctf.ravel()`
Returns flattened CTF tensor.
#### `ctf.reshape()`
Reshapes the input tensor.
#### `ctf.transpose()`
Permute the dimensions of the input tensor.
#### `ctf.tril()` (`ctf.triu()`)
Return lower (upper) triangle of CTF tensor.

### Other operations on CTF Objects
#### `ctf.dot()`
Identical to np.dot (not all functionality supported though).
#### `ctf.einsum()`
Identical to np.einsum (not all functionality supported though).
#### `ctf.qr()`
QR Factorization of input tensor.
#### `ctf.svd()`
Singular Value Decomposition of input tensor
#### `ctf.sum()`
Returns sum of elements in tensor along an axis.
#### `ctf.trace()`
Returns the sum over the diagonal of the input tensor.

### Sparse Functions
#### `ctf.spdiag()`
Sparse Diagonal of input tensor.
#### `ctf.speye()`
Sparse 2D identity tensor.

### Binary Functions
#### `ctf.all()`
#### `ctf.any()`

### MPI Stuff
#### `ctf.MPI_Stop()`
Kills all working nodes.

### Random Module
#### `ctf.random.all_seed()`
Seed the random tensor generator with same seed in all processes.
#### `ctf.random.seed()`
Seed the random tensor generator. 
