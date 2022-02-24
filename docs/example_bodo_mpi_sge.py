#!/usr/bin/env python3
import numpy as np
import bodo


@bodo.jit
def example_function():
    x = np.random.randn(10_000, 10_000, 1)
    y = np.random.randn(10_000, 10_000, 1)
    z = (np.sin(x) + np.cos(y)).sum()
    print(z)


if __name__ == "__main__":
    example_function()