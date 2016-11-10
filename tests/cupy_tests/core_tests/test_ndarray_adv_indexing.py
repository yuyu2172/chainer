import unittest

import numpy

import cupy
from cupy import testing


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'indexes': ([1, -1], slice(None))},
    {'shape': (2, 3, 4), 'indexes': (Ellipsis, [1, 0])},
    {'shape': (2, 3, 4), 'indexes': ([1, -1], Ellipsis)},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), slice(None), [[1, -1], [0, 3]])},
    {'shape': (2, 3, 4), 'indexes': ([1, 0], [2, 1])},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 0], [2, 1])},
    {'shape': (2, 3), 'indexes': ([[0, 1], [1, 0]], [[1, 1], [2, 1]])},
    # array appears with split
    {'shape': (2, 3, 4), 'indexes': ([0, 1], slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'indexes': ([0, 1], slice(None), 1)},
    {'shape': (2, 3, 4), 'indexes': ([1, 0], slice(0, 3, 2), [1, 0])},
    # three arrays
    {'shape': (2, 3, 4), 'indexes': ([1, 0], [2, 1], [3, 1])},
    {'shape': (2, 3, 4), 'indexes': ([1, 0], 1, [3, 1])},
    {'shape': (2, 3, 4), 'indexes': ([1, 0], 1, None, [3, 1])},
    # index broadcasting
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 2], [[1, 0], [0, 1], [-1, 1]])},
)
@testing.gpu
class TestArrayAdvancedIndexingParametrized(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (None, [1, 0], [0, 2], slice(None)),
     'shape': (2, 3, 4), 'indexes': (None, [0, 1], None, [2, 1], slice(None))
     }
)
@testing.gpu
class TestArrayAdvancedIndexingParametrize(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a[self.indexes]

@testing.parameterize(
    {'shape': (2, 3, 4), 'transpose': (1, 2, 0),
     'indexes': (slice(None), [1, 0]),
     'shape': (2, 3, 4), 'transpose': (1, 0, 2),
     'indexes': (None, [1, 2], [0, -1])},
)
@testing.gpu
class TestArrayAdvancedIndexingParametrizedTransp(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.transpose:
            a = a.transpose(self.transpose)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (slice(None), cupy.array([1, 0]))},
    {'shape': (2, 3, 4), 'indexes': (slice(None), numpy.array([1, 0],))},
    {'shape': (2, 3, 4), 'indexes': (cupy.array([1, 0]), numpy.array([0, 1]))},
)
@testing.gpu
class TestArrayAdvancedIndexingArrayClass(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        indexes = list(self.indexes)
        a = testing.shaped_arange(self.shape, xp, dtype)

        if xp is numpy:
            for i, s in enumerate(indexes):
                if isinstance(s, cupy.ndarray):
                    indexes[i] = s.get()

        return a[tuple(indexes)]


@testing.parameterize(
    {'shape': (), 'indexes': ([1],)},
    {'shape': (2, 3), 'indexes': (slice(None), [1, 2], slice(None))},
)
@testing.gpu
class TestArrayInvalidIndexAdv(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_invalid_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        a[self.indexes]
