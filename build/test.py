import numericarray
import numpy as np
a = np.ones(4, dtype=np.float32)
b = np.ones(4, dtype=np.float32)
numericarray.increment(a, b)

print("a", a)
