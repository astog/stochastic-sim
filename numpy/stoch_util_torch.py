from __future__ import print_function
import numpy as np
from itertools import product


def to_stoch(real_n, length, bipolar=True, deterministic=False):
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

    if deterministic:
        np.random.shuffle(stoch_s)

    return stoch_s


def to_real(stoch_s, bipolar=True):
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
        select_s = to_stoch(0.5, stoch_s1.size, bipolar=True, deterministic=True)
        return np.logical_or(np.logical_and(stoch_s1, select_s), np.logical_and(stoch_s2, np.logical_not(select_s))).astype(np.uint8)
    elif mode == 1:
        partial_sum = to_real(stoch_s1, bipolar) + to_real(stoch_s2, bipolar)
        partial_sum = min(partial_sum, 1)
        if bipolar:
            partial_sum = max(partial_sum, -1)
        else:
            partial_sum = max(partial_sum, 0)

        return to_stoch(partial_sum, stoch_s1.size, bipolar, True)
    elif mode == 2:
        if bipolar:
            return np.logical_xor(stoch_s1, stoch_s2).astype(np.uint8)
        else:
            return np.logical_or(stoch_s1, stoch_s2).astype(np.uint8)


def stochify_vector(input_vec, length, bipolar, deterministic):
    '''
    Generates uint8 bit stochastic vector based on input vector.

    @params:
    input_vec       Required:   Input vector 1
    bipolar         Optional:   Whether to use bipolar representation
    deterministic   Optional:   Whether to use deterministic generation
    '''

    size = input_vec.size
    assert(input_vec.shape == (size,))

    stoch_matrix = np.zeros((size, length), dtype=np.uint8)
    if deterministic:
        # Generate number of 1's we want per vector
        num_ones = None
        if bipolar:
            num_ones = (input_vec + 1) * (length / 2)
        else:
            num_ones = input_vec * length

        num_ones = np.round(num_ones).astype(np.uint16).reshape(size, 1)
        col_indices = np.tile(np.arange(length).reshape(1, length), (size, 1))
        stoch_matrix[col_indices < num_ones] = 1

        # shuffle across rows, ie bitstream
        np.apply_along_axis(np.random.shuffle, 1, stoch_matrix)
    else:
        # Generate probabilites for 1's in each bit
        probs = np.random.rand(size, length)
        if bipolar:
            probs = (2 * probs) - 1

        stoch_matrix[probs < input_vec.reshape(size, 1)] = 1

    return stoch_matrix


def binarize_matrix_to_vector(input_matrix, bipolar):
    result = np.zeros(input_matrix.shape[0])
    for i in xrange(input_matrix.shape[0]):
        result[i] = to_real(input_matrix[i], bipolar)

    return result


def stochify_matrix(input_matrix, length, bipolar, deterministic):
    '''
    Generates uint8 bit stochastic matrix based on input matrix.

    @params:
    input_vec       Required:   Input vector 1
    bipolar         Optional:   Whether to use bipolar representation
    deterministic   Optional:   Whether to use deterministic generation
    '''

    assert(input_matrix.shape != (input_matrix.size,))

    rows, cols = input_matrix.shape

    stoch_cube = np.ndarray((rows, cols, length), dtype=np.uint8)
    for (x, y), real_n in np.ndenumerate(input_matrix):
        stoch_cube[x][y] = to_stoch(real_n, length, bipolar, deterministic)

    return stoch_cube


def binarize_cube_to_matrix(input_cube, bipolar):
    result = np.zeros((input_cube.shape[0], input_cube.shape[1]))
    for row, col in product(xrange(input_cube.shape[0]), xrange(input_cube.shape[1])):
        result[row][col] = to_real(input_cube[row][col], bipolar)

    return result


def stoch_dot(input_matrix1, input_matrix2, bipolar, deterministic=False, mode=1):
    '''
    Does dot product but with stochastic streams, hence 2D rather than 1D inputs

    @params:
    input_vec       Required:   Input vector 1
    input_vec       Required:   Input vector 1
    bipolar         Optional:   Whether to use bipolar representation
    '''

    # make sure that dot product is mathematically possible
    assert(input_matrix1.shape == input_matrix2.shape)

    result = to_stoch(0.0, input_matrix1.shape[1], bipolar, deterministic)

    # accumulate product
    for i in xrange(input_matrix1.shape[0]):
        pp = stoch_mul(input_matrix1[i], input_matrix2[i], bipolar)
        # print("{}*{} = {} == {}".format(
        #     to_real(input_matrix1[i], bipolar), to_real(input_matrix2[i], bipolar),
        #     to_real(input_matrix1[i], bipolar) * to_real(input_matrix2[i], bipolar),
        #     to_real(pp, bipolar)), end='\n')
        # print("{} + {} == ".format(to_real(result, bipolar), to_real(pp, bipolar)), end='')
        result = stoch_add(result, pp, bipolar, mode)
        # print(result, end='\n\n')
        # print(to_real(result, bipolar), end='\n\n')

    result = result.astype(np.uint8)
    return result


def stoch_cube_mm(stoch_cube1, stoch_cube2, bipolar=True, deterministic=False, mode=1):
    '''
    Does matrix multiplication but with stochastic streams, hence 3D rather than 2D inputs

    @params:
    stoch_cube1     Required:   Input cube 1
    stoch_cube1     Required:   Input cube 2
    bipolar         Optional:   Whether to use bipolar representation
    '''

    row1, col1, length1 = stoch_cube1.shape
    row2, col2, length2 = stoch_cube2.shape

    # make sure that mm is mathematically possible
    assert(length1 == length2)
    assert(col1 == row2)

    final_cube = np.ndarray((row1, col2, length1), dtype=np.uint8)

    for row, col in product(xrange(row1), xrange(col2)):
        final_cube[row][col] = stoch_dot(stoch_cube1[row][:][:], stoch_cube2[:][col][:], bipolar, deterministic, mode)

    return final_cube


def round_to_fixed_point(vals, bits, bipolar):
    bin_max = None
    bin_min = None
    bin_count = 2**bits - 1
    if bipolar:
        bin_max = (bin_count / 2)
        bin_min = -(bin_count / 2)
        bin_count = bin_max
    else:
        bin_max = bin_count
        bin_min = bin_min

    binary_val = np.round(vals * bin_count)
    return binary_val / bin_count


if __name__ == '__main__':

    bipolar = True
    deterministic = True

    xv = 2 * np.random.rand(20) - 1
    yv = 2 * np.random.rand(20) - 1
    print("\n=========================\n")
    print(xv)
    print(yv)
    # xv = round_to_fixed_point(xv, 8, bipolar)
    # yv = round_to_fixed_point(yv, 8, bipolar)
    # print("\n=========================\n")
    # print(xv)
    # print(yv)
    xm = stochify_vector(xv, 8, bipolar, deterministic)
    ym = stochify_vector(yv, 8, bipolar, deterministic)
    print("\n=========================\n")
    print(binarize_matrix_to_vector(xm, bipolar))
    print(binarize_matrix_to_vector(ym, bipolar))
    print("\n=========================\n")
    ap_xyd = to_real(stoch_dot(xm, ym, bipolar), deterministic)
    print(ap_xyd)
    print(np.dot(xv, yv))
