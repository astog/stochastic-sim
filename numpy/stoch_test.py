import numpy as np
from stoch_util import stochify_vector, stoch_dot, to_real


xv = 2 * np.random.rand(4) - 1
yv = 2 * np.random.rand(4) - 1
xyd = np.dot(xv, yv)
xsm = stochify_vector(xv, 4096, True, True)
ysm = stochify_vector(yv, 4096, True, True)
ap_xyd = to_real(stoch_dot(xsm, ysm, True, True, 0), True)
print(xyd)
print(ap_xyd)
