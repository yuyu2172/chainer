import unittest

import itertools
import numpy

import cupy
from cupy import testing


def perm(iterable):
    return list(itertools.permutations(iterable))
