#!/usr/bin/env python
# coding: utf-8

# # Built-in functions

# <table align="left">
# 
#   <td>
#     <a href="https://colab.research.google.com/github/lukeconibear/swd6_hpp/blob/main/docs/02_build-it_functions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#   </td>
# 
# </table>

# ___
# 
# [Built-in functions](https://docs.python.org/3/library/functions.html) are optimised in C (statically typed and compiled).

# ## [`len`](https://docs.python.org/3/library/functions.html#len)

# In[1]:


nums = [num for num in range(1_000_000)]


# In[2]:


get_ipython().run_cell_magic('timeit', '', 'count = 0\nfor num in nums: # time O(n)\n    count += 1')


# In[3]:


get_ipython().run_line_magic('timeit', 'len(nums) # time O(1)')


# In[ ]:




