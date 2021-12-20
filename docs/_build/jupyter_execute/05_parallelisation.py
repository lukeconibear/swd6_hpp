#!/usr/bin/env python
# coding: utf-8

# # Parallelisation

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukeconibear/swd6_hpp/blob/main/docs/06_parallelisation.ipynb)

# In[1]:


import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    get_ipython().system('pip install dask[dataframe] joblib ray')


# ## What is it?
# 
# Parallelisation divides a large problem into many smaller ones and solves them *simultaneously*.
# - *Divides up the time/space complexity across workers.*
# - Tasks centrally managed by a scheduler.
# - Multi-processing (cores)
#     - Useful for compute-bound problems.
#     - Overcomes the [Global Interpreter Lock, GIL](https://wiki.python.org/moin/GlobalInterpreterLock) (prevents running the bytecode on mutliple threads simutaneously).  
#     - Lower performance when need to exchange/aggregate data
# - Multi-threading (parts of processes)
#     - Useful for memory-bound problems.  
#     
# Parallelised code often introduces overheads. So, the speed-up benefits are more pronounced with bigger jobs, rather than some of the small examples used in this tutorial.

# ## Parallelising a Python?
# 
# Python itself is not designed for massive scalability and controls threads preemptively using the GIL. This has lead many libraries to work around this using C/C++ backends.  
# 
# Some options include:  

# [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) for creating a pool of asynchronous workers. 

# In[2]:


from multiprocessing import Pool

def my_function(x):
    return x * x

with Pool(3) as workers:
    print(workers.map(my_function, [1, 2, 3]))


# [joblib](https://joblib.readthedocs.io/en/latest/) for creating lightweight pipelines that help with "embaressingly parallel" tasks.

# In[3]:


import joblib
import math


# In[4]:


joblib.Parallel(n_jobs=1)(
    joblib.delayed(math.sqrt)(i**2) for i in range(10)
)


# [asyncio](https://docs.python.org/3/library/asyncio.html) for concurrent programs, especially ones that are IO-bound.  
# 
# ```python
# import asyncio
# 
# async def main():
#     print('Hello ...')
#     await asyncio.sleep(1)
#     print('... World!')
#     
# asyncio.run(main())
# ```

# These options work well for the CPU cores on your machine, though not really beyond that.  

# ## [Vectorisation](https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html)
# [Vectors](https://en.wikipedia.org/wiki/Automatic_vectorization) effectively "parallelise" the code by operating on multiple array elements at once, rather than looping through them one at a time.  
# Under the hood for NumPy arrays, functions, and aggregations (e.g., mean, sum).

# In[5]:


import numpy as np


# In[6]:


nums = np.arange(1_000_000)


# In[7]:


get_ipython().run_cell_magic('timeit', '', '[num + 2 for num in nums]')


# In[8]:


get_ipython().run_cell_magic('timeit', '', 'nums + 2 # adds 2 to every elements by overloading the +')


# In[9]:


get_ipython().run_cell_magic('timeit', '', 'np.add(nums, 2)')


# [Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) (operations with different shaped arrays, [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html), [xarray](https://xarray.pydata.org/en/v0.16.2/computation.html?highlight=Broadcasting#broadcasting-by-dimension-name)).

# ![broadcasting.png](images/broadcasting.png)  
# 
# *[Image source](https://mathematica.stackexchange.com/questions/99171/how-to-implement-the-general-array-broadcasting-method-from-numpy)*

# In[10]:


nums_col = np.array([0, 10, 20, 30]).reshape(4, 1)
nums_row = np.array([0, 1, 2])

nums_col + nums_row


# In[11]:


import xarray as xr


# In[12]:


nums_col = xr.DataArray([0, 10, 20, 30], [('col', [0, 10, 20, 30])])
nums_row = xr.DataArray([0, 1, 2], [('row', [0, 1, 2])])

nums_col + nums_row


# NumPy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html) (universal functions).
# - *Optimised in C (statically typed and compiled).*
# - Arbitrary Python function to NumPy ufunc:
#     - [`np.frompyfunc`](https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html). 
#     - [`np.vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html) (keeps the documentation).  

# In[13]:


random_array = np.random.rand(5, 5)
random_array


# In[14]:


def my_function(array, threshold):
    """Compare an array to a threshold."""
    if array > threshold:
        return round(array - threshold, 1)
    else:
        return round(array + threshold, 1)


# In[15]:


frompyfunc_function = np.frompyfunc(
    my_function, 
    2, # number of input arguments 
    1) # number of returned objects
frompyfunc_function(random_array, 0.5)


# In[16]:


frompyfunc_function.__doc__


# In[17]:


vectorized_function = np.vectorize(my_function)
vectorized_function(random_array, 0.5)


# In[18]:


vectorized_function.__doc__


# ## [Dask](https://docs.dask.org/en/latest/)
# 
# - Great features.
# - Helpful documentation.
# - Familiar API.
# - Under the hood for many libraries e.g. [xarray](http://xarray.pydata.org/en/stable/dask.html), [iris](https://scitools.org.uk/iris/docs/v2.4.0/userguide/real_and_lazy_data.html), [scikit-learn](https://ml.dask.org/).

# ### [Single machine](https://docs.dask.org/en/latest/setup/single-distributed.html)
# 
# See the excellent video from Dask creator, Matthew Rocklin, below.

# In[19]:


from IPython.display import IFrame
IFrame(src='https://www.youtube.com/embed/ods97a5Pzw0', width='560', height='315')


# In[20]:


if not IN_COLAB:
    from dask.distributed import Client
    client = Client()
    client 


# If want multiple threads, then could use keyword arguments in Client instance:
# ```python
# client = Client(processes=False, threads_per_worker=4, n_workers=1)
# ```

# Remember (important), always need to close down the client at the end:
# ```python
# client.close()
# ```

# ### Dask behind the scenes

# In[21]:


import xarray as xr


# In[22]:


ds = xr.tutorial.open_dataset(
    'air_temperature',
    chunks={'time': 'auto'} # dask chunks
)


# In[23]:


ds_mean = ds.mean()
ds_mean # a dask.array (an unexecuted task graph)


# In[24]:


ds_mean.compute()


# In[25]:


ds.close()


# ### [dask.array](https://examples.dask.org/array.html) (NumPy)
# See the excellent video from Dask creator, Matthew Rocklin, below.

# In[26]:


from IPython.display import IFrame
IFrame(src='https://www.youtube.com/embed/ZrP-QTxwwnU', width='560', height='315')


# In[27]:


import dask.array as da


# In[28]:


my_array = da.random.random(
    (5_000, 5_000),
    chunks=(500, 500) # dask chunks
)
result = my_array + my_array.T
result


# In[29]:


if not IN_COLAB:
    result.compute()


# ### [dask.dataframe](https://examples.dask.org/dataframe.html) (Pandas)
# See the excellent video from Dask creator, Matthew Rocklin, below.

# In[30]:


IFrame(src='https://www.youtube.com/embed/6qwlDc959b0', width='560', height='315')


# In[31]:


import dask


# In[32]:


df = dask.datasets.timeseries()
df


# In[33]:


type(df)


# In[34]:


result = df.groupby('name').x.std()
result


# In[35]:


# result.visualize()


# In[36]:


result_computed = result.compute()


# In[37]:


type(result_computed)


# ### [dask.bag](https://examples.dask.org/bag.html)
# Iterate over a bag of independent objects (embarrassingly parallel).

# In[38]:


import numpy as np
import dask.bag as db


# In[39]:


nums = np.random.randint(low=0, high=100, size=(5_000))
nums


# In[40]:


def function(nums):
    return chr(nums)


# In[41]:


if not IN_COLAB:
    bag = db.from_sequence(nums)
    bag = bag.map(function)
    
    bag.visualize()
    
    result = bag.compute()
    
    client.close()


# ### [Dask on HPC](https://docs.dask.org/en/latest/setup/hpc.html)
# 
# - Create/edit the [`dask_on_hpc.py`](https://github.com/lukeconibear/swd6_hpp/blob/main/docs/dask_on_hpc.py) file.
# - Submit to the queue using [`qsub dask_on_hpc.bash`](https://github.com/lukeconibear/swd6_hpp/blob/main/docs/dask_on_hpc.bash).
# 
# If need to share memory across chunks:  
# - Use [shared memory](https://docs.dask.org/en/latest/shared.html) (commonly OpenMP, Open Multi-Processing).
# - `-pe smp np` on ARC4
# 
# Otherwise:  
# - Use [message passing interface, MPI](https://docs.dask.org/en/latest/setup/hpc.html?highlight=mpi#using-mpi) (commonly OpenMPI).
# - `-pe ib np` on ARC4

# ## [Ray](https://www.ray.io/)
# Ray will automatically detect the available GPUs and CPUs on the machine.
# - Can also [specify required resources](https://docs.ray.io/en/latest/walkthrough.html#specifying-required-resources).  

# First, initialise Ray.

# In[42]:


import ray
ray.init()


# ### Functions become Tasks
# - Parallelise functions by adding `@ray.remote` decorator  
# - Then instead of calling it normally, use the `.remote()` method  
# - This yields a future object reference that you can retrieve with `ray.get(object)` 

# In[43]:


@ray.remote
def f(x):
    return x * x


# In[44]:


# asynchronously run a task
futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))


# ### Classes become Actors
# - Parallelise classes the same way
# - These actors maintain their internal state  

# In[45]:


@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
        
    def increment(self):
        self.value += 1
    
    def read(self):
        return self.value


# In[46]:


# construct an actor instance using .remote()
counters = [Counter.remote() for i in range(4)]


# In[47]:


# asynchronously run actor methods
[counter.increment.remote() for counter in counters]
futures = [counter.read.remote() for counter in counters]
print(ray.get(futures))


# Other key API methods:
# - `ray.put()`
#     - Put a value in the distributed object store.
#     - `put_id = ray.put(my_object)`
# - `ray.get()`
#     - Get an object from the distributed object store, either placed there by `ray.put()` explicitly or by a task or actor method, blocking until object is available.
#     - `thing = ray.get(put_id)`
# - `ray.wait()`
#     - Wait on a list of ids until one of the corresponding objects is available (e.g., the task completes). Return two lists, one with ids for the available objects and the other with ids for the still-running tasks or method calls.
#     `finished, running = ray.wait([train_id, track_id])`

# ### Ray's [`multiprocessing`](https://docs.ray.io/en/latest/multiprocessing.html)
# To scale beyond one machine and generally manage a pool of processes.  
# 
# Replace:
# ```python
# from multiprocessing.pool import Pool
# ```
# 
# With:
# 
# ```python
# from ray.util.multiprocessing.pool import Pool
# ```
# 

# In[48]:


from ray.util.multiprocessing.pool import Pool


# In[49]:


def my_function(x):
    return x * x


# In[50]:


with Pool(2) as workers:
    print(workers.map(my_function, [1, 2, 3]))


# ### Ray's [`joblib`](https://docs.ray.io/en/latest/joblib.html)
# The underpinnings of [scikit-learn](https://scikit-learn.org/stable/), which Ray can scale to a cluster.
# 
# Import and instantiate `register_ray`, which registers Ray as a `joblib` backend for `scikit-learn`:  
# ```python
# import joblib
# from ray.util.joblib import register_ray
# register_ray()
# ```
# 
# Then run your original `scikit-learn` code within a Ray/`joblib` backend:
# ```python
# with joblib.parallel_backend('ray'):
#     # original scikit-learn code
# ```
# 
# For example, here's some parallel hyperparameter tuning:
# ```python
# import joblib
# from ray.util.joblib import register_ray
# register_ray()
# 
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.svm import SVC
# from sklearn.model_selection import RandomizedSearchCV
# 
# digits = load_digits()
# param_space = {
#     'C': np.logspace(-6, 6, 30),
#     'gamma': np.logspace(-8, 8, 30),
#     'tol': np.logspace(-4, -1, 30),
#     'class_weight': [None, 'balanced'],
# }
# model = SVC(kernel='rbf')
# search = sklearn.model_selection.RandomizedSearchCV(
#     model, param_space, cv=5, n_iter=300, verbose=10)
# 
# with joblib.parallel_backend('ray'):
#     search.fit(digits.data, digits.target)
# ```

# When finished, remember to shut down the Ray connection.

# In[51]:


ray.shutdown()


# Please see this [repository](https://github.com/lukeconibear/distributed_deep_learning) for examples of how to do distributed deep learning using Ray Train with TensorFlow, PyTorch, and Horovod.

# ## [Dask on Ray](https://docs.ray.io/en/latest/data/dask-on-ray.html)
# Use Ray as a backend for Dask tasks.  
# Dask dispatches tasks to Ray for scheduling and execution.

# In[52]:


import ray
import dask
import dask.dataframe as dd 
import pandas as pd
import numpy as np
from ray.util.dask import ray_dask_get


# In[53]:


dask.config.set(scheduler=ray_dask_get) 
ray.init()


# In[54]:


df = pd.DataFrame(np.random.randint(0, 100, size=(2**10, 2**8)))
df = dd.from_pandas(df, npartitions=10)
df.head(10)


# In[55]:


ray.shutdown()


# ## [Modin](https://modin.readthedocs.io/en/latest/)
# Modin uses Ray or Dask to easily speed up your Pandas code.  
# To use Modin, simply replace the import and use Pandas API as normal.

# In[56]:


import os
os.environ['MODIN_ENGINE'] = 'ray'
# os.environ['MODIN_ENGINE'] = 'dask'


# In[57]:


if not IN_COLAB:
    # import pandas as pd
    import modin.pandas as pd

    frame_data = np.random.randint(0, 100, size=(5_000, 1_000))
    df = pd.DataFrame(frame_data)
    df.head(10)


# ## [Mars](https://docs.pymars.org/en/latest/)
# Mars is a tensor-based unified framework for large-scale data computation which scales numpy, pandas, scikit-learn and many other libraries.  
# Swap out the library import, use the same API, and add `.execute()`.

# ```python
# import mars
# mars.new_session()
# ```

# ### [Mars Tensor](https://docs.pymars.org/en/latest/getting_started/tensor.html) for NumPy

# ```python
# # import numpy as np
# # np.random.rand(10)
# 
# import mars.tensor as mt
# mt.random.rand(10).execute()
# ```

# ### [Mars DataFrame](https://docs.pymars.org/en/latest/getting_started/dataframe.html) for Pandas

# ```python
# # import pandas as pd
# # df = pd.DataFrame(
# #     np.random.rand(10),
# #     columns=['random_numbers']
# # )
# 
# import mars.dataframe as md
# df = md.DataFrame(
#     np.random.rand(10),
#     columns=['random_numbers']
# ).execute()
# ```

# And remember to stop the server when you're finished.

# ```python
# mars.stop_server()
# ```

# Mars can also use Ray as the backend ([instructions](https://docs.ray.io/en/latest/data/mars-on-ray.html)).

# ## Further information
# - [Using IPython for parallel computing](https://ipyparallel.readthedocs.io/en/latest/)
# - [Spark on Ray](https://docs.ray.io/en/latest/data/raydp.html): RayDP combines your Spark and Ray clusters, making it easy to do large scale data processing using the PySpark API and seemlessly use that data to train your models using TensorFlow and PyTorch.
# - [Concurrency](https://youtu.be/18B1pznaU1o) can also run different tasks together, but work is not done at the same time.  
# - [Asynchronous](https://youtu.be/iG6fr81xHKA) (multi-threading), useful for massive scaling, threads controlled explicitly.  