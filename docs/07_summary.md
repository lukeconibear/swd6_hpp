# Summary

In this workshop, we covered:

1. [x] Understand how to profile Python code and identify bottlenecks
    - [x] _Measure the time of cells, functions, and programs to find bottlenecks e.g., using [`timeit`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit) and [`line_profiler`](https://github.com/pyutils/line_profiler)._
    - [x] _Visualise the profiled code e.g., using [`SnakeViz`](https://jiffyclub.github.io/snakeviz/) and [`pyinstrument`](https://github.com/joerick/pyinstrument/)._
    - [x] _Log profiling information e.g., using [`Eliot`](https://eliot.readthedocs.io/en/stable/index.html)._
    - [x] _Consider how fast the code could go e.g., Big O notation._
2. [x] Understand how to choose the most appropriate data structure, algorithm, and libraries for a problem
    - [x] _Make use of the built-in functions e.g, use `len` rather than counting the items in an object in a loop._
    - [x] _Use appropriate data structures e.g., append to lists rather than concatenating, use dictionaries as fast to search look-ups, cache results in dictionaries to reduce repeated calculations._
    - [x] _Make use of the standard library (optimised in C) e.g., the `math` module._
    - [x] _See whether there is an algorithm or library that already optimally solves your problem e.g., faster sorting algorithms_.
3. [x] Improve the execution time of Python code using:  
    - [x] Vectorisation
        - [x] _Make use of broadcasting and create your own vectorised functions e.g., with [NumPy](https://numpy.org/doc/stable/reference/ufuncs.html) ufuncs._
    - [x] Compilers
        - [x] _Speed up customs functions e.g., with [Numba](http://numba.pydata.org/) `@njit`._
    - [x] Parallelisation
        - [x] _Distribute work across a single machine or cores on a high-performance computer e.g., with [Dask](https://docs.dask.org/en/latest/) and [Ray](https://www.ray.io/)._
    - [x] GPUs  
        - [x] _Offload functions to the GPU e.g., with [CUDA and Numba](https://developer.nvidia.com/how-to-cuda-python)_.
        - [x] _Accelerate data science workloads e.g., with [RAPIDS](https://developer.nvidia.com/rapids)._
        - [x] _...[JAX](https://jax.readthedocs.io/en/latest/index.html)..._
4. [x] Understand when to use each technique
    - [x] ...