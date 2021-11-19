#!/usr/bin/env python
# coding: utf-8

# # Profiling
# 
# [Profiling](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html) analyses your code in terms of speed and/or memory. This can help identify where the bottlenecks are and how much potential there is for improvement.

# ## [timeit](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit)
# [IPython magic](https://jakevdp.github.io/PythonDataScienceHandbook/01.03-magic-commands.html) are very useful for common problems in data analysis.

# These have 1 `%` at the start for a single line:

# In[1]:


get_ipython().run_line_magic('timeit', 'range(100)')


# Or two `%%` at the start for a cell:

# In[2]:


get_ipython().run_cell_magic('timeit', '', 'for x in range(100):\n    pass')


# ## [line_profiler](https://github.com/pyutils/line_profiler)
# 
# The line profiler tool measures the time spent in each line of a function.

# In[3]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[4]:


def my_maths(x, y):
    z = x * y**2
    print('Gary')
    return z


# In[5]:


get_ipython().run_line_magic('lprun', '-f my_maths my_maths(5, 10)')


# ## [memory_profiler](https://github.com/pythonprofilers/memory_profiler)

# In[6]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[7]:


get_ipython().run_line_magic('memit', 'my_maths(5, 10)')


# For a line-by-line description of memory, you can use `%mprun`, though the function needs to be from a file.

# In[8]:


get_ipython().run_cell_magic('file', 'mprun_example.py', '\ndef my_func():\n    a = [1] * (10**6)\n    b = [2] * (2 * 10**7)\n    del b\n    return a')


# In[9]:


from mprun_example import my_func
get_ipython().run_line_magic('mprun', '-f my_func my_func()')


# ## [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/)
# 
# Pyinstrument is a similar profiler, that helps you find the bottlenecks.

# In[10]:


import time
get_ipython().run_line_magic('load_ext', 'pyinstrument')


# In[11]:


get_ipython().run_cell_magic('pyinstrument', '', 'def a():\n    b()\n    c()\ndef b():\n    d()\ndef c():\n    d()\ndef d():\n    e()\ndef e():\n    time.sleep(1)\na()')


# ## Further information
# [Visual profiler](https://docs.dask.org/en/latest/diagnostics-local.html#example) for parallel code using Dask.  
# [PythonSpeed.com](https://pythonspeed.com/), Itamar Turner-Trauring.  
