#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=00:15:00
#$ -pe smp 1
#$ -l h_vmem=1G

# ensure linear algebra libraries using 1 thread
# https://docs.dask.org/en/stable/array-best-practices.html#avoid-oversubscribing-threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

conda activate parallelisation_tests

python example_dask_jobqueue_sge.py