#!/usr/bin/env python
# coding: utf-8

# # Parallelisation

# ## What is it?
# 
# Parallelisation divides a large problem into many smaller ones and solves them *simultaneously*.
# - *Divides up the time/space complexity across workers.*
# - Tasks centrally managed by a scheduler.
# - Multi-processing (cores) - useful for compute-bound problems.
# - Multi-threading (parts of processes) - useful for memory-bound problems.

# ## Parallelising a Python?
# 
# Python itself is not designed for massive scalability and controls threads preemptively using a [Global Interpreter Lock, GIL](https://wiki.python.org/moin/GlobalInterpreterLock). This has lead many libraries to work around this using C/C++ backends. Some options include:
# - [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) for creating a pool of asynchronous workers.  
# - [joblib](https://joblib.readthedocs.io/en/latest/) for creating lightweight pipelines.  
# - [asyncio](https://docs.python.org/3/library/asyncio.html) for concurrent programs.  
# 
# These options work well for the CPU cores on your machine, though not really beyond that.  

# ## Moving over to HPC
# 
# If need to share memory across chunks:  
# - Use [shared memory](https://docs.dask.org/en/latest/shared.html) (commonly OpenMP, Open Multi-Processing).
# - `-pe smp np` on ARC4
# 
# Otherwise:  
# - Use [message passing interface, MPI](https://docs.dask.org/en/latest/setup/hpc.html?highlight=mpi#using-mpi) (commonly OpenMPI).
# - `-pe ib np` on ARC4

# In[ ]:





# In[ ]:




