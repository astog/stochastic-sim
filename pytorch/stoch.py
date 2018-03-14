from __future__ import print_function
import torch

import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def quantize(tensor, nbits=8):
    # one bit is needed for the sign
    return tensor.mul(2**(nbits - 1)).round().div(2**(nbits - 1))


def to_stoch(float_tensor, length, bipolar=True, deterministic=False):
    '''
    params:
    float_tensor    Tensor between [0,1] or [-1,1] based on setting of bipolar
    length          Length of the stochastic bit stream
    bipolar         Set to false if the input tensor is between [0, 1]

    returns:        Returns a byte tensor one higher dim than input, with the last dim being the bit array
    '''

    if deterministic:
        # Find the number of 0's we want in our stream
        num_zeros = None
        if bipolar:
            num_zeros = (float_tensor + 1) * (length / 2.0)
        else:
            num_zeros = float_tensor * length

        # Setup for broadcasting
        mask_shape = float_tensor.size() + (length,)
        num_zeros = num_zeros.round().type(torch.LongTensor).view(float_tensor.size() + (1,))

        # The mask values are a random permutation of arange(0, length-1)
        mask_vals = torch.randperm(length).type(torch.LongTensor)
        # The stochastic stream is just the compare mask since it returns ByteTensor of appropriate shape (due to broadcast)
        return mask_vals >= num_zeros
    else:
        # Add dimension to input so broadcasting to mask_shape works
        mask_shape = float_tensor.size() + (length,)
        float_tensor = float_tensor.view(float_tensor.size() + (1,))

        # Compute random numbers based on unipolar/bipolar
        samples = torch.rand(*mask_shape)
        if bipolar:
            samples = (2 * samples) - 1

        return samples >= float_tensor


def to_real(stoch_tensor, bipolar=True):
    '''
    params:
    stoch_tensor    The stochastic bitstream byte-array
    bipolar         Set to false if the stochastic stream represnts a float between [0, 1]. True if [-1, 1]

    returns:
    real_n          Floating point value approximation, made by this stream
    '''

    length = stoch_tensor.shape[-1]

    # Make sure to cast before sum since sum with ByteTensor results in overflow for longer lengths
    num_ones = stoch_tensor.type(torch.FloatTensor).sum(-1)

    if bipolar:
        return (1 - (2 * num_ones / length))

    return num_ones / length


def multiply(stoch_tensor1, stoch_tensor2, bipolar=True):
    '''
    params:
    stoch_tensor1   Stochastic tensor 1
    stoch_tensor1   Stochastic tensor 2, needs to be the same shape as stoch_tensor1
    bipolar         Set to true if tensor 1, 2 are b/w [1, 1]

    returns:
    The stochastic multiplication between these two streams
    '''

    if bipolar:
        return stoch_tensor1 ^ stoch_tensor2
    else:
        return stoch_tensor1 & stoch_tensor2


def sum(stoch_vector, bipolar=True, deterministic=False, mode='deter'):
    '''
    params:
    stoch_vector    Vector of stochastic streams
    bipolar         Set to true if tensor 1, 2 are b/w [1, 1]
    mode            Mode to do addition: deter, stoch

    returns:
    Stochastic sum of the streams in stoch_vector
    '''

    length = stoch_vector.shape[-1]

    if mode == 'deter':
        accum_sum = to_real(stoch_vector, bipolar).sum(dim=0)
        if bipolar:
            accum_sum.clamp_(-1, 1)
        else:
            accum_sum.clamp_(0, 1)
        return to_stoch(accum_sum, length, bipolar=bipolar, deterministic=deterministic)

    elif mode == 'stoch':
        # TODO: Implement this
        return sum(stoch_vector, bipolar=bipolar, deterministic=deterministic)


def dot(stoch_vector1, stoch_vector2, bipolar=True, output_stoch=True):
    '''
    params:
    stoch_vector1   Vector of stochastic streams
    stoch_vector2   Vector of stochastic streams. Same size as stoch_vector1
    bipolar         Set to true if tensor 1, 2 are b/w [1, 1]
    output_stoch    Whether the output should be a stochastic stream or a non-clipped real number

    returns:
    Stochastic stream of the dot product result
    '''

    # Make sure it is a vector of stochastic streams
    assert(len(stoch_vector1.shape) == 2)
    assert(len(stoch_vector2.shape) == 2)

    mul_result = multiply(stoch_vector1, stoch_vector2, bipolar)
    if output_stoch:
        return sum(mul_result, bipolar)
    else:
        return to_real(mul_result).sum()


def mm(stoch_matrix1, stoch_matrix2, bipolar=True, deterministic=False, output_stoch=True):
    '''
    params:
    stoch_matrix1   Matrix of stochastic streams
    stoch_matrix2   Matrix of stochastic streams. Same size as stoch_vector1
    bipolar         Set to true if tensor 1, 2 are b/w [1, 1]
    output_stoch    Whether the output should be a stochastic stream or a non-clipped real number

    returns:
    Returns the stochastic equivalent to matrix multiplication
    '''

    # Make sure it is a matrix of stochastic streams
    assert(len(stoch_matrix1.shape) == 3)
    assert(len(stoch_matrix2.shape) == 3)

    # Make sure matrix multiplication is mathematically valid
    assert(stoch_matrix1.shape[1] == stoch_matrix2.shape[0])
    assert(stoch_matrix1.shape[2] == stoch_matrix2.shape[2])

    length = stoch_matrix1.shape[2]

    m1 = stoch_matrix1.view(stoch_matrix1.shape[0], stoch_matrix1.shape[1], 1, length)
    m2 = stoch_matrix2.view(1, stoch_matrix2.shape[0], stoch_matrix2.shape[1], length)
    dot_sum = torch.sum(to_real(multiply(m1, m2, bipolar), bipolar), dim=1)

    if output_stoch:
        return to_stoch(dot_sum, length, bipolar=bipolar, deterministic=deterministic)
    else:
        return dot_sum


if __name__ == '__main__':
    deterministic = False
    bipolar = True
    nbits = 8
    length = 32

    size = 4096

    x = None
    y = None
    if bipolar:
        x = torch.FloatTensor(1, size).normal_(0.0, 0.25).clamp_(-1, 1)
        y = torch.FloatTensor(size, size).normal_(0.0, 0.25).clamp_(-1, 1)
    else:
        x = torch.FloatTensor(1, size).normal_(0.5, 0.125).clamp_(0, 1)
        y = torch.FloatTensor(size, size).normal_(0.5, 0.125).clamp_(0, 1)

    x = quantize(x, nbits)
    y = quantize(y, nbits)
    # print(x)
    # print(y)

    apx = to_stoch(x, length, bipolar=bipolar, deterministic=deterministic)
    apy = to_stoch(y, length, bipolar=bipolar, deterministic=deterministic)
    # print(to_real(apx, bipolar))
    # print(to_real(apy, bipolar))

    print("Ideal ====== ")
    print(torch.mm(x, y), end='\n\n')
    print("Stoch Ideal ------")
    ideal_sol = torch.mm(to_real(apx, bipolar), to_real(apy, bipolar))
    print(ideal_sol, end='\n\n')
    print("Stoch Actual ~~~~~")
    got_sol = mm(apx, apy, bipolar, output_stoch=False)
    print(got_sol, end='\n\n')
    datos = np.reshape((ideal_sol - got_sol).numpy(), size)
    # best fit of data
    (mu, sigma) = norm.fit(datos)

    # the histogram of the data
    n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    # plot
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.grid(True)

    plt.show()
