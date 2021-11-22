#!/usr/bin/env python
# coding: utf-8

# # [Built-in functions](https://docs.python.org/3/library/functions.html)
# Optimised in C (statically typed and compiled).

# ## [`len`](https://docs.python.org/3/library/functions.html#len)

# In[1]:


nums = [num for num in range(1_000_000)]


# In[2]:


get_ipython().run_cell_magic('timeit', '', 'count = 0\nfor num in nums: # time O(n)\n    count += 1')


# In[3]:


get_ipython().run_line_magic('timeit', 'len(nums) # time O(1)')


# In[ ]:





# In[ ]:




