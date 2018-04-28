import time
import torch
import numpy as np
import numericarray

a = np.ones(2 * 1000000, dtype=np.float32)
b = np.ones(2 * 1000000, dtype=np.float32)

print("adding with numpy:")

t1 = time.time()
for i in range(100):
    a += b
print("time = ", time.time() - t1)

print("adding with pytorch")

at = torch.from_numpy(a)
bt = torch.from_numpy(b)

t1 = time.time()
for i in range(100):
    a += b
print("time = ", time.time() - t1)

print("adding with numericarray")

t1 = time.time()
for i in range(100):
    numericarray.increment(a, b)
print("time = ", time.time() - t1)
