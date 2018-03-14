from __future__ import print_function
import torch


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

    stoch_s = None
    if deterministic:
        # Find the number of 1's we want in our stream
        num_ones = None
        if bipolar:
            num_ones = (float_tensor + 1) * (length / 2)
        else:
            num_ones = float_tensor * length

        # Setup for broadcasting
        mask_shape = float_tensor.size() + (length,)
        num_ones = torch.round(num_ones).type(torch.LongTensor).view(float_tensor.size() + (1,))

        # The mask values are a random permutation of arange(0, length-1)
        mask_vals = torch.randperm(length)

        # Set the bits we want
        stoch_s = torch.zeros(*mask_shape)
        stoch_s[mask_vals < num_ones] = 1
    else:
        # Add dimension to input so broadcasting to mask_shape works
        mask_shape = float_tensor.size() + (length,)
        float_tensor = float_tensor.view(float_tensor.size() + (1,))

        # Compute random numbers based on unipolar/bipolar
        probs = torch.rand(*mask_shape)
        if bipolar:
            probs = (2 * probs) - 1

        # Set the bits we want
        stoch_s = torch.zeros(*mask_shape)
        stoch_s[probs < float_tensor] = 1

    return stoch_s


def to_real(stoch_tensor, bipolar=True):
    '''
    params:
    stoch_tensor    The stochastic bitstream byte-array
    bipolar         Set to false if the stochastic stream represnts a float between [0, 1]. True if [-1, 1]

    returns:
    real_n          Floating point value approximation, made by this stream
    '''

    length = stoch_tensor.shape[-1]
    num_ones = stoch_tensor.sum(-1).type(torch.FloatTensor)

    if bipolar:
        return (2 * num_ones / length) - 1

    return num_ones / length


if __name__ == '__main__':
    bipolar = True
    nbits = 8
    length = 16

    x = torch.rand(5)
    if bipolar:
        x = 2*x -1

    x = quantize(x, nbits)
    print(x)

    apx = to_stoch(x, length, bipolar=bipolar, deterministic=True)
    # print(apx)
    print(to_real(apx, bipolar=bipolar))
