from __future__ import print_function
import numpy as np
from itertools import product
from stoch_util import *


def stochify_vector(input_vec, length, bipolar, deterministic):
    '''
    Generates uint8 bit stochastic vector based on input vector.

    @params:
    input_vec       Required:   Input vector 1
    bipolar         Optional:   Whether to use bipolar representation
    deterministic   Optional:   Whether to use deterministic generation
    '''

    assert(input_vec.shape == (input_vec.size,))

    # we shuffle if deterministic generation
    shuffle = deterministic

    stoch_matrix = np.ndarray((input_vec.size, length), dtype=np.uint8)
    for i, real_n in np.ndenumerate(input_vec):
        stoch_matrix[i] = toStochS(real_n, length, bipolar, deterministic, shuffle)

    return stoch_matrix


def binarize_matrix_to_vector(input_matrix, bipolar):
    result = np.zeros(input_matrix.shape[0])
    for i in xrange(input_matrix.shape[0]):
        result[i] = toRealN(input_matrix[i], bipolar)

    return result


def stochify_matrix(input_matrix, length, bipolar, deterministic):
    '''
    Generates uint8 bit stochastic matrix based on input vector.

    @params:
    input_vec       Required:   Input vector 1
    bipolar         Optional:   Whether to use bipolar representation
    deterministic   Optional:   Whether to use deterministic generation
    '''

    assert(input_matrix.shape != (input_matrix.size,))

    # we shuffle if deterministic generation
    shuffle = deterministic

    rows, cols = input_matrix.shape

    stoch_cube = np.ndarray((rows, cols, length), dtype=np.uint8)
    for (x, y), real_n in np.ndenumerate(input_matrix):
        stoch_cube[x][y] = toStochS(real_n, length, bipolar, deterministic, shuffle)

    return stoch_cube


def binarize_cube_to_matrix(input_cube, bipolar):
    result = np.zeros((input_cube.shape[0], input_cube.shape[1]))
    for row, col in product(xrange(input_cube.shape[0]), xrange(input_cube.shape[1])):
        result[row][col] = toRealN(input_cube[row][col], bipolar)

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

    result = toStochS(0.0, input_matrix1.shape[1], bipolar, deterministic, deterministic)

    # accumulate product
    for i in xrange(input_matrix1.shape[0]):
        pp = stoch_mul(input_matrix1[i], input_matrix2[i], bipolar)
        # print("{}*{} = {} == {}".format(
        #     toRealN(input_matrix1[i], bipolar), toRealN(input_matrix2[i], bipolar),
        #     toRealN(input_matrix1[i], bipolar) * toRealN(input_matrix2[i], bipolar),
        #     toRealN(pp, bipolar)), end='\n\n')
        print(pp)
        print(result)
        # print("{} + {} == ".format(toRealN(result, bipolar), toRealN(pp, bipolar)), end='\n')
        # result = stoch_add(result, pp, bipolar, mode)
        print(result, end='\n\n')
        # print(toRealN(result, bipolar))

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
    ap_xyd = toRealN(stoch_dot(xm, ym, bipolar), deterministic)
    print(ap_xyd)
    print(np.dot(xv, yv))
