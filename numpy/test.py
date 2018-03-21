import torch
import numpy as np
from stoch_nn import *

x = 2*np.random.rand(100000) - 1
xm = stochify_vector(x, 128, True, False)
y = 2*np.random.rand(100000) - 1
ym = stochify_vector(x, 128, True, False)

print(stoch_dot(xm, ym, True))
