import unittest

import numpy

import cupy
from cupy import get_array_module
from cupy import testing


def wrap_scatter(a, ind, v, axis=None, mode=''):
    if get_array_module(a) is numpy:
        if mode == 'update':
            axis %= a.ndim
            ind = [slice(None)] * axis +\
                [ind] + [slice(None)] * (a.ndim - axis - 1)
            a[ind] = v
    else:
        if mode == 'update':
            a.scatter_update(ind, v, axis)


def compute_v_shape(in_shape, indices_shape, axis):
    if len(in_shape) == 0:
        return indices_shape
    else:
        axis %= len(in_shape)
        lshape = in_shape[:axis]
        rshape = in_shape[axis + 1:]
    return lshape + indices_shape + rshape


@testing.parameterize(
    *testing.product({
        'indices': [2, [0, 1], -1, [-1, -2], [[1, 0], [2, 3]]],
        'axis': [0, 1, 2, -1, -2],
    })
)
@testing.gpu
class TestScatterUpdate(unittest.TestCase):

    shape = (4, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scatter_update(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        v_shape = compute_v_shape(
            self.shape, numpy.array(self.indices).shape, self.axis)
        v = testing.shaped_arange(v_shape, xp, dtype)
        wrap_scatter(a, self.indices, v, self.axis, mode='update')
        return a


@testing.parameterize(
    {'shape': (3, 4, 5), 'indices': [0, 1], 'axis': 0,
     'v': numpy.array([1]), 'cupy': True},
    {'shape': (3, 4, 5), 'indices': [[0, 1], [2, 3]], 'axis': 1,
     'v': numpy.arange(5), 'cupy': True},
    {'shape': (3, 4, 5), 'indices': [0, 1], 'axis': 0,
     'v': numpy.array([1]), 'cupy': False},
    {'shape': (3, 4, 5), 'indices': [0, 1], 'axis': 1,
     'v': numpy.array(1), 'cupy': False},
    {'shape': (3, 4, 5), 'indices': [[0, 1], [2, 3]], 'axis': 1,
     'v': 1, 'cupy': False},
    {'shape': (3,), 'indices': [0, 1], 'axis': 0,
     'v': numpy.array([1, 2]), 'cupy': True}
)
@testing.gpu
class TestScatterUpdateParamterized(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scatter_update(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        v = self.v
        if isinstance(v, numpy.ndarray):
            v = self.v.astype(dtype)
        if self.cupy and xp == cupy:
            v = cupy.array(v)
        wrap_scatter(a, self.indices, v, self.axis, mode='update')
        return a


@testing.parameterize(
    *testing.product({
        'shape': [(3, 4, 5)],
        'indices_shape': [(2,)],
        'axis': [1],
        'v_shape': [(2, 3), (3,)],
        'mode': ['update'],
    })
)
@testing.gpu
class TestScatterOpErrorMismatch(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_shape_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.float32)
        i = testing.shaped_arange(self.indices_shape, xp, numpy.int32) % 3
        v = testing.shaped_arange(self.v_shape, xp, numpy.float32)
        wrap_scatter(a, i, v, self.axis, self.mode)
