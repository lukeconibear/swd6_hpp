#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=00:10:00
#$ -pe smp 8
#$ -l h_vmem=24G

module load intel openmpi

# stop processes pinning to one core
unset GOMP_CPU_AFFINITY

# ensure linear algebra libraries using 1 thread
# https://docs.dask.org/en/stable/array-best-practices.html#avoid-oversubscribing-threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

conda activate swd6_hpp

# ensure that number of cores here match that in the requested resources at the top
mpiexec -n 8 python example_bodo_mpi_sge.py