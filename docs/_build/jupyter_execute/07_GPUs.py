#!/usr/bin/env python
# coding: utf-8

# # GPUs
# - *GPUs (Graphics Processing Units) are optimised for numerical operations, while CPUs perform general computation.*
# - [cuPy](https://cupy.dev/) (NumPy).
# ```python
# import cupy as cp
# x = cp.arange(6).reshape(2, 3).astype('f')
# x.sum(axis=1)
# ```
# - [cuDF](https://docs.rapids.ai/api/cudf/stable/) (Pandas).
# - Can run in [parallel](https://docs.dask.org/en/latest/gpu.html).
#   - GPUs available on [ARC4](https://arcdocs.leeds.ac.uk/systems/arc4.html#using-the-v100-gpu-nodes).

# In[ ]:




