import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _sampler_type = libcudnn.CUDNN_SAMPLER_BILINEAR


class SpatialTfGrid(function.Function):

    def __init__(self, out_height, out_width, use_cudnn=True):
        self.out_height
        self.out_width
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 2)

        #x_type = in_types[0]
        #grid_type = in_types[1]
        #type_check.expect(
        #    x_type.dtype.kind == 'f',
        #    grid_type.dtype.kind == 'f',
        #    x_type.ndim == 4,
        #    grid_type.ndim == 4,
        #    x_type.shape[0] == grid_type.shape[0],
        #)

    def forward_cpu(self, inputs):
        theta, = inputs

    def forward_gpu(self, inputs):
        theta, = inputs
        grid_t = cuda.cupy.empty(
            self.out_shape[:1] + (self.out_height, self.out_width) + (2,),
            dtype=theta.dtype)
        if cuda.cudnn_enabled and self.use_cudnn and _cudnn_version >= 5000:
            shape = numpy.array(self.out_shape, dtype=numpy.int32)
            theta = cuda.cupy.ascontiguousarray(theta)
            handle = cudnn.get_handle()
            self.st_desc =\
                cuda.cupy.cudnn.create_spatial_transformer_descriptor(
                    _sampler_type, grid_t.dtype, len(shape), shape.ctypes.data)

            libcudnn.spatialTfGridGeneratorForward(
                handle, self.st_desc.value, theta.data.ptr, grid_t.data.ptr)
            grid = cuda.cupy.transpose(grid_t, (0, 3, 1, 2))

        return grid,

    def backward_gpu(self, inputs, grad_outputs):
        theta, = inputs
        ggrid, = grad_outputs
        ggrid_t = cuda.cupy.transpose(ggrid, (0, 2, 3, 1))

        gtheta = cuda.cupy.empty_like(theta)
        if cuda.cudnn_enabled and self.use_cudnn and _cudnn_version >= 5000:
            handle = cudnn.get_handle()
            ggrid_t = cuda.cupy.ascontiguousarray(ggrid_t)
            libcudnn.spatialTfGridGeneratorBackward(
                handle, self.st_desc.value, ggrid_t.data.ptr, gtheta.data.ptr)
        return gtheta,


def spatial_tf_grid(theta, out_height, out_width, use_cudnn=True):
    """

    Args:
        theta (~chainer.Variable):  (B, 2, 3)
    Returns:
        grid (~chainer.Variable):  (B, 2, H, W).
            In the 2nd dimension, the first coordinate is u,
            and the second coordinate is v.
    """
    return SpatialTfGrid(out_height, out_width, use_cudnn)(theta)
