#!/bin/sh

g++ Wave2D.cpp Timer.cpp -o Wave2D_mine


mpic++ Wave2D_mpi.cpp Timer.cpp -fopenmp -o Wave2D_mpi

