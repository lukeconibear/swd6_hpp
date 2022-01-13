#!/usr/bin/env python
# coding: utf-8

# # Data structures and algorithms

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukeconibear/swd6_hpp/blob/main/docs/03_data_structures_and_algorithms.ipynb)

# In[1]:


# if you're using colab, then install the required modules
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    get_ipython().system('pip install algorithms')


# ## [Data structures](https://docs.python.org/3/library/datatypes.html)
# - [Common](https://docs.python.org/3/tutorial/datastructures.html) and [additional data structures](https://docs.python.org/3/library/collections.html).

# ### [Lists](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
# - Append to lists, rather than concatenating.  
# - *Lists are allocated twice the memory required, so appending fills this up in O(1) (long-term average), while concatenating creates a new list each time in O(n).*
# 

# In[2]:


get_ipython().run_cell_magic('timeit', '', 'my_list = []\nfor num in range(1_000):\n    my_list += [num] # time O(n)')


# In[3]:


get_ipython().run_cell_magic('timeit', '', 'my_list = []\nfor num in range(1_000):\n    my_list.append(num) # time O(1)')


# ### [Dictionaries](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
# - Fast to search, O(1).  
# 
# *Example from Luciano Ramalho, [Fluent Python, Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python/9781491946237/), 2015. O'Reilly Media, Inc.*

# In[4]:


import numpy as np


# In[5]:


haystack_list = np.random.uniform(low=0, high=100, size=(1_000_000))

haystack_dict = {key: value for key, value in enumerate(haystack_list)}

needles = [0.1, 50.1, 99.1]


# In[6]:


get_ipython().run_cell_magic('timeit', '', 'needles_found = 0\nfor needle in needles:\n    if needle in haystack_list: # time O(n) within list\n        needles_found += 1')


# In[7]:


get_ipython().run_cell_magic('timeit', '', 'needles_found = 0\nfor needle in needles:\n    if needle in haystack_dict: # time O(1) within dict\n        needles_found += 1')


# - Many more [examples](https://www.bigocheatsheet.com/) e.g.:
#   - Generators save memory by yielding only the next iteration.
#   - Memory usage for floats/integers of 16 bit < 32 bit < 64 bit.
#   - For NetCDFs, using [`engine='h5netcdf'`](https://github.com/shoyer/h5netcdf) with `xarray` can be faster, over than the default [`engine='netcdf4'`](https://github.com/Unidata/netcdf4-python).  
#   - [Compression](https://youtu.be/8pFnrr0NnwY): If arrays are mostly 0, then can save memory using [sparse arrays](https://sparse.pydata.org/en/stable/quickstart.html).
#   - [Chunking](https://youtu.be/8pFnrr0NnwY): If need all data, then can load/process in chunks to reduce amount in memory: [Zarr](https://zarr.readthedocs.io/en/stable/) for arrays, [Pandas](https://pythonspeed.com/articles/chunking-pandas/).
#   - [Indexing](https://youtu.be/8pFnrr0NnwY): If need a subset of the data, then can index (multi-index) to reduce memory and increase speed for queries: [Pandas](https://pythonspeed.com/articles/indexing-pandas-sqlite/), [SQLite](https://docs.python.org/3/library/sqlite3.html).

# ### Reduce repeated calculations with [caching](https://realpython.com/lru-cache-python/)
# - e.g. [Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_number) (each number is the sum of the two preceding ones starting from 0 and 1 e.g. 0, 1, 1, 2, 3, 5, 8, 13, 21, 34).

# In[8]:


def fibonacci(n): # time O(2^n) as 2 calls to the function n times (a balanced tree of repeated calls)
    if n == 0 or n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


# In[9]:


get_ipython().run_line_magic('timeit', 'fibonacci(20)')


# In[10]:


def fibonacci_with_caching(n, cache={0: 0, 1: 0, 2: 1}): # time O(n) as 1 call per n
    if n in cache:
        return cache[n]
    else:
        cache[n] = fibonacci_with_caching(n - 1, cache) + fibonacci_with_caching(n - 2, cache)
        return cache[n]


# In[11]:


get_ipython().run_line_magic('timeit', 'fibonacci_with_caching(20, cache={0: 0, 1: 0, 2: 1})')


# ### [Lazy loading](https://xarray.pydata.org/en/v0.16.2/dask.html) and [execution](https://tutorial.dask.org/01x_lazy.html)
# - Lazily loads metadata only, rather than eagerly loading data into memory.
# - Creates task graph of scheduled work awaiting execution (`.compute()`).

# In[12]:


import xarray as xr


# In[13]:


xr.tutorial.open_dataset('air_temperature')


# ### [Loop-invariants](https://en.wikipedia.org/wiki/Loop_invariant)
# Move them *outside* the loop.  
# 
# Loops are slow in Python ([CPython](https://www.python.org/), default interpreter), because loops type−check and dispatch functions per cycle. Try to avoid them where can (e.g., using NumPy ufuncs, aggregations, broadcasting, etc.). 

# In[14]:


get_ipython().run_cell_magic('timeit', '', 'for num in range(1_000_000):\n    constant = 500_000\n    bigger_num = max(num, constant)')


# In[15]:


get_ipython().run_cell_magic('timeit', '', 'constant = 500_000\nfor num in range(1_000_000):\n    bigger_num = max(num, constant)')


# ### Suitable data types for parallel computing
# - [Parquet](https://examples.dask.org/dataframes/01-data-access.html#Write-to-Parquet) creates efficient tabular data (e.g. dataframes), useful for parallel computing.
#   - Accessed via [pyarrow](https://arrow.apache.org/docs/python/install.html).
# - [Zarr](https://zarr.readthedocs.io/en/stable/) creates compressed, chunked, N-dimensional arrays, designed for use in parallel computing.

# ## Algorithms
# - The instructions to solve the problem.
#   - Free MIT course on '*Introduction to algorithms*', [video lectures](https://youtube.com/playlist?list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb).
# - Many existing libraries are already optimised (computationally and algorithmically).
#   - [Minimal examples of data structures and algorithms in Python](https://github.com/keon/algorithms).

# In[16]:


from algorithms import sort


# In[17]:


unsorted_array = np.random.rand(1_000)


# [Selection sort](https://en.wikipedia.org/wiki/Selection_sort)  
# *Time O(n^2), space O(1)*
# 
# 1. Two arrays: one unsorted (original) and one sorted (can do in place to avoid extra memory).
# 2. Find smallest number in the unsorted array and add to the sorted array.
# 3. Repeat step 2 until final index is the largest number (i.e. sorted array).

# In[18]:


get_ipython().run_line_magic('timeit', 'sort.selection_sort(unsorted_array)')


# [Merge sort](https://en.wikipedia.org/wiki/Merge_sort)  
# *Time O(nlgn), space O(n or nlgn, depending on implementation)*
# 
# 1. Divide array in half.  
# 2. Recusively apply:  
#     a. Step 1 to both halves, until hit the base case where both halves are length 1.  
#     b. Merge the two length 1 arrays into sorted array of length 2.  
#     c. Repeat b, all the way up for this half.  

# In[19]:


get_ipython().run_line_magic('timeit', 'sort.merge_sort(unsorted_array)')


# ## Further information
# [LeetCode](https://leetcode.com)  
# [AlgoExpert](https://www.algoexpert.io/product)