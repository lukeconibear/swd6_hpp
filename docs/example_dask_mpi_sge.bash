#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=00:15:00
#$ -pe ib 8
#$ -l h_vmem=1G
#$ -P admiralty

module load intel openmpi

# ensure linear algebra libraries using 1 thread
# https://docs.dask.org/en/stable/array-best-practices.html#avoid-oversubscribing-threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

conda activate parallelisation_tests

# ensure that number of cores here match that in the requested resources at the top
mpirun -np 8 python example_dask_mpi_sge.py