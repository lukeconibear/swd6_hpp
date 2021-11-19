#!/usr/bin/env python
# coding: utf-8

# # Profiling
# 
# [Profiling](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html) analyses your in terms of speed and/or memory. This can help identify where the bottlenecks are and how much potential there is for improvement.

# ## [line_profiler](https://github.com/pyutils/line_profiler)
# 
# - [IPython magic](https://jakevdp.github.io/PythonDataScienceHandbook/01.03-magic-commands.html) (Jupyter Lab)
#   - Line: `%timeit`
#   - Cell: `%%timeit`
#   - If `pip install line_profiler`:
#     - First load module: `%load_ext line_profiler`
#     - Scripts: `%prun`
#     - Line-by-line: `%lprun`
#       - `@profile` decorator around the function

# ## [memory_profiler](https://github.com/pythonprofilers/memory_profiler)
# 
# - If `pip install memory_profiler`:
# - First load module:
#   - `%load_ext memory_profiler`
#   - Line: `%memit`
#   - Cell: `%%memit`
#   - Line-by-line: `%mprun`
# 

# ## [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/)
# ...

# #### Profile parallel code
# - [Example visualisation of profiler](https://docs.dask.org/en/latest/diagnostics-local.html#example).

# ## Further information
# [PythonSpeed.com](https://pythonspeed.com/), Itamar Turner-Trauring

# In[ ]:




