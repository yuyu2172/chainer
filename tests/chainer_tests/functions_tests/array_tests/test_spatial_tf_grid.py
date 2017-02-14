import unittest

import numpy as np

from chainer import gradient_check
from chainer import testing
from chainer import cuda
from chainer.testing import attr

from chainer.functions.array.spatial_tf_grid import spatial_tf_grid, SpatialTfGrid


class TestSpatialTfGrid(unittest.TestCase):

    B = 3

    def setUp(self):
        self.theta = np.stack([np.array([[1, 0, 0, 0, 1, 0]], dtype=np.float32)] * self.B)
        self.output_shape = (self.B, 4, 5, 5)
        self.grads = np.random.uniform(
            size=(self.B,) + (2,) + self.output_shape[2:]).astype(self.theta.dtype)

    def check_forward(self, theta, output_shape):
        stg = spatial_tf_grid(theta, output_shape).data
        #print (stg + 1.) / 2. * 4


    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.theta), self.output_shape)

#    def check_backward(self, theta, output_shape, grads):
#        gradient_check.check_backward(
#            SpatialTfGrid(output_shape), (theta,), (grads,))
#
#    @attr.gpu
#    def test_backward_gpu(self):
#        self.check_backward(cuda.to_gpu(self.theta),
#                            self.output_shape,
#                            cuda.to_gpu(self.grads))
#
