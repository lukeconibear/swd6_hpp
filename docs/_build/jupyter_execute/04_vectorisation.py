#!/usr/bin/env python
# coding: utf-8

# # Vectorisation

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukeconibear/swd6_hpp/blob/main/docs/04_vectorisation.ipynb)

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

# In[3]:


import numpy as np


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


import xarray as xr


# In[9]:


nums_col = xr.DataArray([0, 10, 20, 30], [('col', [0, 10, 20, 30])])
nums_row = xr.DataArray([0, 1, 2], [('row', [0, 1, 2])])

nums_col + nums_row


# - NumPy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html) (universal functions).
#   - *Optimised in C (statically typed and compiled).*
#   - Arbitrary Python function to NumPy ufunc:
#       - [`np.frompyfunc`](https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html). 
#       - [`np.vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html).  

# In[10]:


random_array = np.random.rand(5, 5)
random_array


# In[11]:


def my_function(array, threshold):
    """Compare an array to a threshold."""
    if array > threshold:
        return round(array - threshold, 1)
    else:
        return round(array + threshold, 1)


# In[12]:


frompyfunc_function = np.frompyfunc(
    my_function, 
    2, # number of input arguments 
    1) # number of returned objects
frompyfunc_function(random_array, 0.5)


# In[13]:


frompyfunc_function.__doc__


# In[14]:


vectorized_function = np.vectorize(my_function)
vectorized_function(random_array, 0.5)


# In[15]:


vectorized_function.__doc__

