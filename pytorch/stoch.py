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


def multiply(stoch_tensor1, stoch_tensor2, bipolar=True):
    '''
    params:
    stoch_tensor1   Stochastic tensor 1
    stoch_tensor1   Stochastic tensor 2, needs to be the same shape as stoch_tensor1
    bipolar         Set to true if tensor 1, 2 are b/w [1, 1]

    returns:
    The stochastic multiplication between these two streams
    '''

    assert(stoch_tensor1.shape == stoch_tensor2.shape)

    if bipolar:
        return stoch_tensor1 ^ stoch_tensor2
    else:
        return stoch_tensor1 & stoch_tensor2


def sum(stoch_vector, bipolar=True, mode='deter'):
    '''
    params:
    stoch_vector    Vector of stochastic streams
    bipolar         Set to true if tensor 1, 2 are b/w [1, 1]
    mode            Mode to do addition: deter, stoch

    returns:
    Stochastic stream of the sum
    '''

    length = stoch_vector.shape[-1]

    if mode == 'deter':
        accum_sum = to_real(stoch_vector, bipolar).sum(dim=0)
        if bipolar:
            accum_sum.clamp_(-1, 1)
        else:
            accum_sum.clamp_(0, 1)
        return to_stoch(accum_sum, length, bipolar)

    elif mode == 'stoch':
        # TODO: Implement this
        return sum(stoch_vector, bipolar)


def dot(stoch_vector1, stoch_vector2, bipolar=True):
    '''
    params:
    stoch_vector1   Vector of stochastic streams
    stoch_vector2   Vector of stochastic streams. Same size as stoch_vector1
    bipolar         Set to true if tensor 1, 2 are b/w [1, 1]

    returns:
    Stochastic stream of the dot product result
    '''

    # Make sure it is a vector of stochastic streams
    assert(len(stoch_vector1.shape) == 2)
    assert(len(stoch_vector2.shape) == 2)

    return sum(multiply(stoch_vector1, stoch_vector2, bipolar), bipolar)


if __name__ == '__main__':
    deterministic = True
    bipolar = True
    nbits = 8
    length = 16

    x = torch.rand(5)
    y = torch.rand(5)
    if bipolar:
        x = 2 * x - 1
        y = 2 * y - 1

    x = quantize(x, nbits)
    y = quantize(x, nbits)

    apx = to_stoch(x, length, bipolar=bipolar, deterministic=deterministic)
    apy = to_stoch(y, length, bipolar=bipolar, deterministic=deterministic)

    print("Ideal ====== ")
    print(torch.dot(x, y), end='\n\n')
    print("Stoch Ideal ------")
    print(torch.dot(to_real(apx, bipolar), to_real(apy, bipolar)), end='\n\n')
    print("Stoch Actual ~~~~~")
    print(to_real(dot(apx, apy, bipolar), bipolar), end='\n\n')
