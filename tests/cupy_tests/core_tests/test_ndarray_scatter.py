import unittest

import numpy

from cupy import core
from cupy import get_array_module
from cupy import testing


def scatter_add_cpu(a, ind, v, axis=None):
    axis %= a.ndim
    ind = numpy.array(ind)
    v_shape = a.shape[:axis] + ind.shape + a.shape[axis+1:]
    v_br = numpy.broadcast_to(v, v_shape)

    v_flat = v_br.reshape(a.shape[:axis] + (numpy.prod(ind.shape),) +
                           a.shape[axis+1:])
    ind_flat = ind.flatten()

    for i, ind in enumerate(ind_flat):
        a_slices = ([slice(None)] * axis + [ind] +
                    [slice(None)] * (a.ndim - axis - 1))
        v_slices = ([slice(None)] * axis + [i] +
                    [slice(None)] * (a.ndim - axis - 1))
        a[a_slices] += v_flat[v_slices]


def wrap_scatter(a, ind, v, axis=None, mode=''):
    if get_array_module(a) is numpy:
        if mode == 'update':
            axis %= a.ndim
            ind = ([slice(None)] * axis +
                        [ind] + [slice(None)] * (a.ndim - axis - 1))
            a[ind] = v
        elif mode == 'add':
            scatter_add_cpu(a, ind, v, axis)
    else:
        if mode == 'update':
            a.scatter_update(ind, v, axis)
        elif mode == 'add':
            a.scatter_add(ind, v, axis)


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
    {'indices_shape': (2,), 'axis': 0, 'v_shape': (1,)},
    {'indices_shape': (2, 2), 'axis': 1, 'v_shape': (5,)},
)
@testing.gpu
class TestScatterUpdateParamterized(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scatter_update(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        m = a.shape[self.axis]
        indices = testing.shaped_arange(
            self.indices_shape, xp, numpy.int32) % m
        v = testing.shaped_arange(self.v_shape, xp, dtype=dtype)
        wrap_scatter(a, indices, v, self.axis, mode='update')
        return a


@testing.parameterize(
    *testing.product({
        'indices': [2, [0, 1], -1, [-1, -2], [[1, 1], [2 , 1]]],
        'axis': [0, 1, 2, -1, -2],
        'dtype': [numpy.float32, numpy.int32]
    })
)
@testing.gpu
class TestScatterAdd(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.numpy_cupy_array_equal()
    def test_scatter_op(self, xp):
        a = xp.zeros(self.shape, dtype=self.dtype)
        v_shape = compute_v_shape(
            self.shape, numpy.array(self.indices).shape, self.axis)
        v = testing.shaped_arange(v_shape, xp, self.dtype)
        wrap_scatter(a, self.indices, v, self.axis, mode='add')
        return a


@testing.parameterize(
    {'indices_shape': (2,), 'axis': 0, 'v_shape': (1,)},
    {'indices_shape': (2, 2), 'axis': 0, 'v_shape': (5,)},
    {'indices_shape': (2, 2), 'axis': 1, 'v_shape': (5,)},
    {'indices_shape': (2, 2, 3), 'axis': 1, 'v_shape': (2, 2, 3, 5)},
)
@testing.gpu
class TestScatterAddParamterized(unittest.TestCase):

    shape = (3, 4, 5)
    dtype = numpy.float32

    @testing.numpy_cupy_array_equal()
    def test_scatter_add(self, xp):
        a = xp.zeros(self.shape, dtype=self.dtype)
        m = a.shape[self.axis]
        indices = testing.shaped_arange(
            self.indices_shape, xp, numpy.int32) % m
        v = testing.shaped_arange(self.v_shape, xp, dtype=self.dtype)
        wrap_scatter(a, indices, v, self.axis, mode='add')
        return a


@testing.parameterize(
    *testing.product({
        'shape': [(3, 4, 5)],
        'indices': [(2,)],
        'axis': [1],
        'v_shape': [(2, 3), (3,)],
        'mode': ['update', 'add'],
    })
)
@testing.gpu
class TestScatterOpErrorMismatch(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_shape_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.float32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
        v = testing.shaped_arange(self.v_shape, xp, numpy.float32)
        wrap_scatter_update(a, i, v, self.axis)
