import unittest

import itertools
import numpy

import cupy
from cupy import testing


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 0], [0, 2]), 'value': 1.},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [0, 1],  [2, 1]), 'value': 1}
)
@testing.gpu
class TestScatterMultiArraySetitem(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scatter_multi_array_setitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        value = xp.array(self.value, dtype=dtype)
        a[self.indexes] = self.value
        return a
