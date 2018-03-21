from __future__ import print_function
import torch
import numpy as np


def quantize(tensor, nbits=8):
    # one bit is needed for the sign
    return tensor.mul(2**(nbits - 1)).round().div(2**(nbits - 1))


def binarize(float_tensor, bipolar=True):
    '''
    params:
    float_tensor    Tensor between [0,1] or [-1,1] based on setting of bipolar
    bipolar         Set to false if the input tensor is between [0, 1]

    returns:        Returns a byte tensor of the same shape as float_tensor. Each byte element contains 1 samples of that probability
    '''

    if bipolar:
        # inverse bipolar represenetation
        return float_tensor.new(*float_tensor.size()).uniform_(-1, 1) >= float_tensor
    else:
        return float_tensor.new(*float_tensor.size()).uniform_(0, 1) < float_tensor


def to_real(sample_sum_tensor, total_samples, bipolar=True, dtype=torch.FloatTensor):
    '''
    params:
    sample_sum_tensor   The sum of various samples ()
    total_samples
    bipolar             Set to false if the stochastic stream represnts a float between [0, 1]. True if [-1, 1]

    returns:
    real_n          Floating point value approximation, made by this stream
    '''

    if bipolar:
        sample_sum_tensor = ((total_samples / 2) - sample_sum_tensor)
        return sample_sum_tensor.type(dtype) / (total_samples / 2)
    else:
        return sample_sum_tensor.type(dtype) / total_samples


def to_samples(float_tensor, total_samples, bipolar=True):
    sum_tensor = torch.IntTensor(*float_tensor.size()).zero_()
    for i in xrange(total_samples):
        sum_tensor.add_(binarize(float_tensor, bipolar))

    return sum_tensor


def to_stoch_byte(float_tensor, bipolar=True):
    stoch_tensor = torch.ByteTensor(*float_tensor.size()).zero_()
    for i in xrange(8):
        stoch_tensor.add_((2**i) * binarize(float_tensor, bipolar))

    return stoch_tensor


if __name__ == '__main__':
    x = torch.tanh(torch.randn(5))
    apx = to_stoch_byte(x)
    print(x)
    print(apx)
    for num in apx.numpy():
        print(np.unpackbits(num))
