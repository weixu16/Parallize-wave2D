# Parallize-wave2D

This program parallelizes a sequential version of a two-dimensional wave diffusion program, using a hybrid form of MPI and OpenMPI. The formula was based on Schroedinger's Wave Dissemination.


## Parallize Strategy
#### MPI:

Allocate each rank a whole z[3][size][size] size;

When at time t0, t1, each rank calculates the whole z[0], z[1];

From time t2, each rank only calculates its own part, rank*size/mpi_size, after calculating, exchange its boundary information to its left and right neighbors;

When exchanging information, the odd ranks send their left and right information first, and receive later; the even ranks receive first, and send later. That will avoid dead lock.

If needs printing, ranks which are not 0, send its data to rank 0.

#### Open MP:

When each rank is calculating, parallelize the calculation each step. After each step is done, when each rank is exchanging information, it will not be parallelized.


## Performance Evaluation
The performance improves 4.73 times with 4 machines, each has 4 multi-threads compares to sequential program


## Limitations and Possible Performance Improvement
#### Limitations:  When printing out the result, the performance is not improved.
#### Performance Improvement: 
Use one of the threads to print, other threads still can calculate.

For all the ranks, use one thread to deal with the boundary information exchanging and then calculate the boundaries, other threads can work on unboundary calcs.





