import numpy as np
from numba import cuda

@cuda.jit
def add_arrays(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

n = 1000000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.empty_like(a)

threadsperblock = 256
blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

add_arrays[blockspergrid, threadsperblock](a, b, c)

print("Sum of arrays:", c[:5])
