import numpy
from six import moves

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
    if _cudnn_version >= 4000:
        _bwd_filter_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        _bwd_data_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT


def check_input_changed(input_x):
    """
    check if input shape is different from the input shape taken previously.
    """
    assert input_x.flags.c_contiguous





            


class Convolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, use_cudnn=True, cover_all=False,
                 deterministic=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.use_cudnn = use_cudnn
        self.cover_all = cover_all
        self.deterministic = deterministic

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        out_c, _, kh, kw = W.shape
        n, c, h, w = x.shape

        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph,
                                      cover_all=self.cover_all)
        assert out_h > 0, 'Height in the output should be positive.'
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw,
                                      cover_all=self.cover_all)
        assert out_w > 0, 'Width in the output should be positive.'

        y = cuda.cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)
        if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            if b is not None:
                b = cuda.cupy.ascontiguousarray(b)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx), x.dtype)
            if b is not None:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])

            workspace_size = cuda.get_max_workspace_size()
            workspace = cuda.cupy.empty((workspace_size,), dtype='b')
            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, _fwd_pref,
                workspace_size)

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            libcudnn.convolutionForward(
                handle, one.data, x_desc.value, x.data.ptr,
                self.filter_desc.value, W.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero.data,
                y_desc.value, y.data.ptr)

            # TODO(beam2d): Support unshared bias
            if b is not None:
                cudnn.add_tensor(
                    handle, one.data, self.bias_desc.value, b.data.ptr,
                    one.data, y_desc.value, y.data.ptr)

        return y,


def convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True,
                   cover_all=False, deterministic=False):
    """Two-dimensional convolution function.

    This is an implementation of two-dimensional convolution in ConvNets.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_O, c_I, k_H, k_W)`.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
        deterministic (bool): The output of this function can be
            non-deterministic when it uses cuDNN.
            If this option is ``True``, then it forces cuDNN to use
            a deterministic algorithm. This option is only available for
            cuDNN version >= v4.


    Returns:
        ~chainer.Variable: Output variable.

    The two-dimensional convolution function is defined as follows.
    Then the ``Convolution2D`` function computes correlations between filters
    and patches of size :math:`(k_H, k_W)` in ``x``.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``-pad`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Let :math:`(s_Y, s_X)` be the stride of filter application, and
    :math:`(p_H, p_W)` the spatial padding size. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W) / s_X + 1.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    .. seealso:: :class:`~chainer.links.Convolution2D`

    """
    func = Convolution2DFunction(
        stride, pad, use_cudnn, cover_all, deterministic)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
