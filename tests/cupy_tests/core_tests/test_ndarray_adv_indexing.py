import unittest

from cupy import testing


@testing.parameterize(
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'transpose': (1, 2, 0), 'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': ([1, -1], slice(None))},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (Ellipsis, [1, 0])},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': ([1, -1], Ellipsis)},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None), slice(None), [1, -1])},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None), slice(None), [[1, -1], [0, 3]])},
)
@testing.gpu
class TestArrayAdvancedIndexingParametrized(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.transpose:
            a = a.transpose(self.transpose)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (), 'transpose': None, 'indexes': ([1],)},
    {'shape': (2, 3), 'transpose': None,
     'indexes': (slice(None), [1], slice(None))},
)
@testing.gpu
class TestArrayInvalidIndexAdv(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_invalid_adv_getitem(self, xp, dtype):

        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.transpose:
            a = a.transpose(self.transpose)
        a[self.indexes]
