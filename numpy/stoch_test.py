import numpy as np
from stoch_util import stochify_vector


xv = 2 * np.random.rand(5) - 1
xm = stochify_vector(xv, 16, True, True)
print("\n=============\n")
print(xv)
print("\n=============\n")
print(xm)
