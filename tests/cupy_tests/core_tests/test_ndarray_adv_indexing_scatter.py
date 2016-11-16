import unittest

import itertools
import numpy

import cupy
from cupy import core
from cupy import get_array_module
from cupy import testing


def wrap_multi_array_scatter(a, slices, v, mode=''):
    if get_array_module(a) is numpy:
        if mode == 'update':
            a[slices] = v
        elif mode == 'add':
            pass
            # a[slices] += v
    else:
        if mode == 'update':
            a.scatter_multi_array_update(slices, v)
        elif mode == 'add':
            a.scatter_multi_array_add(slices, v)


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 0], [0, 2]), 'value': 1.},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [0, 1],  [2, 1]), 'value': 1}
)
@testing.gpu
class TestScatterMultiArrayUpdate(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scatter_multi_array_update(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        value = xp.array(self.value, dtype=dtype)
        wrap_multi_array_scatter(a, self.indexes, value, mode='update')
        return a


#@testing.parameterize(
#    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 1, 1], [0, 0, 2]), 'value': 1.},
#    {'shape': (2, 3, 4), 'indexes': (slice(None), [0, 1],  [2, 1]), 'value': 1}
#)
#@testing.gpu
#class TestScatterMultiArrayAdd(unittest.TestCase):
#
#    dtype = numpy.float32
#
#    @testing.numpy_cupy_array_equal()
#    def test_scatter_multi_array_update(self, xp):
#        a = testing.shaped_arange(self.shape, xp, self.dtype)
#        value = xp.array(self.value, dtype=self.dtype)
#        wrap_multi_array_scatter(a, self.indexes, value, mode='add')
#        return a
