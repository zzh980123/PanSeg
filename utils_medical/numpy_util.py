import numpy as np


def get_i_dim_slice(np_array: np.ndarray, dim, s: slice):
    ndims = np_array.ndim
    return np_array[(slice(None), ) * dim + s + (slice(None), ) * (ndims - dim)]


def create_i_dim_padding(s, i, ndims):
    remain_dim = ndims - i - 1
    padding_tuple = ((0, 0),) * i + (s,) + ((0, 0),) * remain_dim
    return padding_tuple