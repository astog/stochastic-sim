import numpy as np
from stoch_util import toStochS, toRealN


def get_approx_stats(samples, length, bipolar, deterministic):
    def vec_approx_n(real_n):
        stoch_s = toStochS(real_n, length, bipolar=bipolar,
                           deterministic=deterministic)
        approx_n = toRealN(stoch_s, bipolar=bipolar)
        return approx_n

    vec_approx = np.vectorize(vec_approx_n)

    x = np.random.rand(samples)
    if bipolar:
        x = 2 * x - 1

    y = vec_approx(x)
    delta = x - y

    return (np.average(delta), np.std(delta))
