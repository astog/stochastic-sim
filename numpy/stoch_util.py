import numpy as np


def toStochS(real_n, length, bipolar=True, deterministic=False, shuffle=False):
    '''
    params:
    real_n -    real number between [-1, 1]
    length -    length of the returned array. Higher the length, the better is the approximation
    bipolar -   Set to false if the real_n is between [0, 1] not [-1 1]

    returns:
    stoch_s -   the stochastic bitstream byte-array
    '''
    stoch_s = None

    if deterministic:
        # Find the number of 1's we want in our stream
        n1 = None
        if bipolar:
            n1 = (real_n + 1) * (length / 2)
        else:
            n1 = real_n * length

        n1 = np.round(n1).astype(int)
        n0 = length - n1

        # Minor optimization, set the least required bits
        if n1 < n0:
            stoch_s = np.zeros(length, dtype=np.uint8)
            stoch_s[:n1] = 1
        else:
            stoch_s = np.ones(length, dtype=np.uint8)
            stoch_s[n1:] = 0
    else:
        stoch_s = np.zeros(length, dtype=np.uint8)
        probs = np.random.rand(length)
        if bipolar:
            probs = (2 * probs) - 1
        stoch_s[probs < real_n] = 1

    if shuffle:
        np.random.shuffle(stoch_s)

    return stoch_s


def toRealN(stoch_s, bipolar=True):
    '''
    params:
    stoch_s -   the stochastic bitstream byte-array
    bipolar -   Set to false if the stochastic stream represnts a float between [0, 1]. True if [-1, 1]

    returns:
    real_n -    floating point value approximation, made by this stream
    '''
    n = stoch_s.size
    assert(n > 0)

    n1 = float(stoch_s.sum())
    n0 = n - n1
    if bipolar:
        return (n1 - n0) / n
    else:
        return n1 / n


def stoch_mul(stoch_s1, stoch_s2, bipolar):
    if bipolar:
        return np.logical_not(np.logical_xor(stoch_s1, stoch_s2)).astype(np.uint8)
    else:
        return np.logical_and(stoch_s1, stoch_s2).astype(np.uint8)


def stoch_add(stoch_s1, stoch_s2, bipolar=True, mode=1):
    if mode == 0:
        select_s = toStochS(0.5, stoch_s1.size, bipolar=True, deterministic=True, shuffle=True)
        return np.logical_or(np.logical_and(stoch_s1, select_s), np.logical_and(stoch_s2, np.logical_not(select_s))).astype(np.uint8)
    elif mode == 1:
        return toStochS(toRealN(stoch_s1, bipolar) + toRealN(stoch_s2, bipolar), stoch_s1.size, bipolar, True, True)
    elif mode == 2:
        return np.logical_or(stoch_s1, stoch_s2).astype(np.uint8)
