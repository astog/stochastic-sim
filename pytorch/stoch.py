from __future__ import print_function
import torch


def quantize(tensor, nbits=8):
    # one bit is needed for the sign
    return tensor.mul(2**(nbits - 1)).round().div(2**(nbits - 1))


def to_stoch(float_tensor, length, bipolar=True, deterministic=True):
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
            num_zeros = (float_tensor + 1) * (length / 2)
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
        return (1 - (2 * num_ones / length)).squeeze()

    return (num_ones / length).squeeze()


if __name__ == '__main__':
    deterministic = True
    bipolar = True
    nbits = 8
    length = 1024

    x = torch.rand(5)
    if bipolar:
        x = 2 * x - 1

    x = quantize(x, nbits)
    print(x)

    apx = to_stoch(x, length, bipolar=bipolar, deterministic=deterministic)
    print(apx)
    print(to_real(apx, bipolar=bipolar))
