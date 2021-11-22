#!/usr/bin/env python
# coding: utf-8

# # GPUs

# <table align="left">
# 
#   <td>
#     <a href="https://colab.research.google.com/github/lukeconibear/swd6_hpp/blob/main/docs/01_profiling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#   </td>
# 
# </table>

# GPUs (Graphics Processing Units) are optimised for numerical operations, while CPUs perform general computation.
# - [cuPy](https://cupy.dev/) (NumPy).
# ```python
# import cupy as cp
# x = cp.arange(6).reshape(2, 3).astype('f')
# x.sum(axis=1)
# ```
# - [cuDF](https://docs.rapids.ai/api/cudf/stable/) (Pandas).
# 
# 
# 
# - Can run in [parallel](https://docs.dask.org/en/latest/gpu.html).
#   - GPUs available on [ARC4](https://arcdocs.leeds.ac.uk/systems/arc4.html#using-the-v100-gpu-nodes).

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

# In[ ]:




