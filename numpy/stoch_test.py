import numpy as np
from stoch_util import stochify_vector


xv = 2 * np.random.rand(10000) - 1
xm = stochify_vector(xv, 1024, bipolar=True, deterministic=True)
