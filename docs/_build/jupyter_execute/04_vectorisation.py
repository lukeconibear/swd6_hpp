#!/usr/bin/env python
# coding: utf-8

# # Vectorisation

# ## [Loop-invariants](https://en.wikipedia.org/wiki/Loop_invariant)
# Move them *outside* the loop.

# In[1]:


get_ipython().run_cell_magic('timeit', '', 'for num in range(1_000_000):\n    constant = 500_000\n    bigger_num = max(num, constant)')


# In[2]:


get_ipython().run_cell_magic('timeit', '', 'constant = 500_000\nfor num in range(1_000_000):\n    bigger_num = max(num, constant)')


# ## Use [vectorisation](https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html) instead of loops
# - Loops are slow in Python ([CPython](https://www.python.org/), default interpreter).
#   - *Because loops type−check and dispatch functions per cycle.*
# - [Vectors](https://en.wikipedia.org/wiki/Automatic_vectorization) can work on many parts of the problem at once.
# - NumPy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html) (universal functions).
#   - *Optimised in C (statically typed and compiled).*
#   - [Arbitrary Python function to NumPy ufunc](https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html).

# In[3]:


import numpy as np
import xarray as xr


# In[4]:


nums = np.arange(1_000_000)


# In[5]:


get_ipython().run_cell_magic('timeit', '', 'for num in nums:\n    num *= 2')


# In[6]:


get_ipython().run_cell_magic('timeit', '', 'double_nums = np.multiply(nums, 2)')


# - [Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) (ufuncs over different shaped arrays, [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html), [xarray](https://xarray.pydata.org/en/v0.16.2/computation.html?highlight=Broadcasting#broadcasting-by-dimension-name)).

# ![broadcasting.png](images/broadcasting.png)  
# 
# *[Image source](https://mathematica.stackexchange.com/questions/99171/how-to-implement-the-general-array-broadcasting-method-from-numpy)*

# In[7]:


nums_col = np.array([0, 10, 20, 30]).reshape(4, 1)
nums_row = np.array([0, 1, 2])

nums_col + nums_row


# In[8]:


nums_col = xr.DataArray([0, 10, 20, 30], [('col', [0, 10, 20, 30])])
nums_row = xr.DataArray([0, 1, 2], [('row', [0, 1, 2])])

nums_col + nums_row


# In[ ]:




