#!/usr/bin/env python
# coding: utf-8

# # [Compilers](https://youtu.be/I4nkgJdVZFA)
# - [CPython](https://www.python.org/)
#   - *Ahead-Of-Time (AOT) compiler.*
#     - Statically compiled C extensions.
#   - General purpose interpreter.
#     - Can work on a variety of problems.
#   - Dynamically typed.
#     - Types can change e.g. `x = 5`, then later `x = 'gary'`.
# - [PyPy](https://www.pypy.org/)
#   - *Just−In−Time (JIT) compiler (written in Python).*
#     - Enables optimisations at run time, especially for numerical tasks with repitition and loops.
#     - Replaces CPython.
#     - Faster, though overheads for start-up and memory.
# - [Numba](http://numba.pydata.org/)
#   - *Uses JIT compiler on functions.*
#     - Converts to fast machine code (LLVM).
#     - Uses decorators around functions.
#     - Use with the default CPython.
#     - Examples for [NumPy](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html) and [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#using-numba).

# In[1]:


from numba import njit

nums = np.arange(1_000_000)


# In[ ]:


def super_function(nums):
    trace = 0.0
    for num in nums: # loop
        trace += np.cos(num) # numpy
    return nums + trace # broadcasting


# In[ ]:


get_ipython().run_line_magic('timeit', 'super_function(nums)')


# In[ ]:


@njit # numba decorator
def super_function(nums):
    trace = 0.0
    for num in nums: # loop
        trace += np.cos(num) # numpy
    return nums + trace # broadcasting


# In[ ]:


get_ipython().run_line_magic('timeit', 'super_function(nums)')


# - [Cython](https://cython.org/)
#   - *Compiles to statically typed C/C++*.
#   - Use for any amount of code.
#   - Use with the default CPython.
#   - Examples [not using IPython](https://cython.readthedocs.io/en/latest/src/quickstart/build.html#building-a-cython-module-using-setuptools), [NumPy](https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html), [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html) (example below).

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.DataFrame({
    "a": np.random.randn(1000),
    "b": np.random.randn(1000),
    "N": np.random.randint(100, 1000, (1000)),
    "x": "x",
})
df.head()


# In[ ]:


def f(x):
    return x * (x - 1)
   

def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
        
    return s * dx


# In[ ]:


get_ipython().run_line_magic('timeit', 'df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[ ]:


get_ipython().run_cell_magic('cython', '# only change', 'def f(x):\n    return x * (x - 1)\n   \n\ndef integrate_f(a, b, N):\n    s = 0\n    dx = (b - a) / N\n    for i in range(N):\n        s += f(a + i * dx)\n        \n    return s * dx')


# In[ ]:


get_ipython().run_line_magic('timeit', 'df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)')


# In[ ]:


get_ipython().run_cell_magic('cython', '', 'cdef double f(double x) except? -2:                  # adding types\n    return x * (x - 1)\n   \n\ncpdef double integrate_f(double a, double b, int N): # adding types\n    cdef int i                                       # adding types\n    cdef double s, dx                                # adding types\n    s = 0\n    dx = (b - a) / N\n    for i in range(N):\n        s += f(a + i * dx)\n        \n    return s * dx')


# In[ ]:


get_ipython().run_line_magic('timeit', 'df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)')

