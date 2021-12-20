#!/usr/bin/env python
# coding: utf-8

# # GPUs

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukeconibear/swd6_hpp/blob/main/docs/07_GPUs.ipynb)

# GPUs (Graphics Processing Units) are optimised for numerical operations, while CPUs (central processing units) perform general computation.
# 
# Could use other types of accelerators too.

# ## [JAX](https://jax.readthedocs.io/en/latest/index.html)

# ...

# In[ ]:





# ## Automatic detection

# Many libraries can use GPUs automatically if they can detect one.
# 
# [`TensorFlow`](https://www.tensorflow.org/install/gpu)
# ```python
# import tensorflow as tf
# tf.config.list_physical_devices('GPU')
# ```
# 
# [`PyTorch`](https://pytorch.org/docs/stable/notes/cuda.html)
# ```python
# import torch
# torch.cuda.is_available()
# ```

# ## [CUDA](https://developer.nvidia.com/how-to-cuda-python)

# ...

# In[ ]:





# ## [RAPIDS](https://developer.nvidia.com/rapids)
# Accelerated data science libraries, including cuPy for NumPy and SciPy, cuDF for Pandas, and others such as XGBoost, cuML, cuGraph.

# ### [cuPy](https://cupy.dev/)
# 
# ```python
# # NumPy for CPU
# >>> import numpy as np
# >>> x_cpu = np.zeros((10, ))
# >>> y_cpu = np.zeros((10, 5))
# >>> z_cpu = np.dot(x_cpu, y_cpu)
# >>> z_cpu = cp.asnumpy(z_gpu) # convert over
# 
# # CuPy for GPU
# >>> import cupy as cp
# >>> x_gpu = cp.zeros((10, ))
# >>> y_gpu = cp.zeros((10, 5))
# >>> z_gpu = cp.dot(x_gpu, y_gpu)
# >>> z_gpu = cp.asarray(z_cpu) # convert over
# ```

# ### [cuDF](https://docs.rapids.ai/api/cudf/stable/)
# 
# ```python
# # Pandas for CPU
# >>> import pandas as pd
# >>> df = pd.DataFrame({
#         'a': list(range(20)),
#         'b': list(reversed(range(20))),
#         'c': list(range(20))
#     })
# 
# # cuDF for GPU
# >>> import cudf
# >>> df = cudf.DataFrame({
#         'a': list(range(20)),
#         'b': list(reversed(range(20))),
#         'c': list(range(20))
#     })
# ```

# ## Parallel GPUs
# Can run across multiple GPUs in [parallel](https://docs.dask.org/en/latest/gpu.html).  
# GPUs available on [ARC4](https://arcdocs.leeds.ac.uk/systems/arc4.html#using-the-v100-gpu-nodes).  

# ## Further information
# [CuPy - Sean Farley](https://www.youtube.com/watch?v=_AKDqw6li58), PyBay 2019.  
# [cuDF - Mark Harris](https://www.youtube.com/watch?v=lV7rtDW94do), PyCon AU 2019.  
