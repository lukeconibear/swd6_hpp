#!/usr/bin/env python
# coding: utf-8

# # Profiling

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukeconibear/swd6_hpp/blob/main/docs/01_profiling.ipynb)

# In[1]:


# if you're using colab, then install the required modules
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    get_ipython().system('pip install line_profiler memory_profiler pyinstrument')


# [Profiling](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html) analyses your code in terms of speed and/or memory. This can help identify where the bottlenecks are and how much potential there is for improvement.

# ## [timeit](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit)
# [IPython magic commands](https://jakevdp.github.io/PythonDataScienceHandbook/01.03-magic-commands.html) are very useful for common problems in data analysis.
# 
# `timeit` is one of them, which measures the time execution of an expression. It runs a few times, depending on how intensive the expression is, and returns the average and error. It is useful for benchmarking a small code snippet.
# 
# These have one `%` at the start for a single line:

# In[2]:


get_ipython().run_line_magic('timeit', 'range(100)')


# Or two `%%` at the start for a cell:

# In[3]:


get_ipython().run_cell_magic('timeit', '', 'for x in range(100):\n    pass')


# For profiling longer functions and programs, the Python standard library has:
# - [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile)
# - [`profile`](https://docs.python.org/3/library/profile.html#module-profile)

# In[4]:


def my_function():
    a = [1] * (10**6)
    b = [2] * (2 * 10**7)
    del b
    return a


# In[5]:


import cProfile


# In[6]:


cProfile.run('my_function()')


# Though the `line_profiler` module is a user-friendly alternative.

# ## [line_profiler](https://github.com/pyutils/line_profiler)
# 
# The `line_profiler` module measures the time spent in each line of a function.

# In[7]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[8]:


get_ipython().run_line_magic('lprun', '-f my_function my_function()')


# ## [memory_profiler](https://github.com/pythonprofilers/memory_profiler)
# 
# The `memory_profiler` module measures the memory used by a function, at its peak and the overall increment.

# In[9]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[10]:


get_ipython().run_line_magic('memit', 'my_function()')


# Though sometimes you'll want to see the memory allocations on each line. For this, you can use `%mprun`, though the function needs to be from a file.
# 
# Note, `%%file <filename>` is another IPython magic command, to write the cell contents to disk.

# In[11]:


get_ipython().run_cell_magic('file', 'mprun_example.py', '\ndef my_function():\n    a = [1] * (10**6)\n    b = [2] * (2 * 10**7)\n    del b\n    return a')


# In[12]:


from mprun_example import my_function
get_ipython().run_line_magic('mprun', '-f my_function my_function()')


# ## [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/)
# 
# `pyinstrument` is a statistical profiling module of wall-clock time (recording the call stack every 1ms), lowering the overhead compared to tracing profilers. It hides library frames, so you can focus on the slow parts of your code. The output shows *how* the function executes using a traffic light colour legend.

# In[13]:


import time
get_ipython().run_line_magic('load_ext', 'pyinstrument')


# In[14]:


get_ipython().run_cell_magic('pyinstrument', '', 'def a():\n    b()\n    c()\ndef b():\n    d()\ndef c():\n    d()\ndef d():\n    e()\ndef e():\n    time.sleep(1)\na()')


# ## Further information
# [Visual profiler](https://docs.dask.org/en/latest/diagnostics-local.html#example) for parallel code using Dask.  
# [PythonSpeed.com](https://pythonspeed.com/), Itamar Turner-Trauring.  
