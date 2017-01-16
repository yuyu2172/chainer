import math

from chainer import cuda
from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import link
import cupy





def get_algo_family():
    return ['...']

warmup_iterations = 5

class Finder(object):

    def __init__(device):
        self.id = device
        self.reset_algorithm_cache()
        self.iteration = 0

    def reset_algorithm_cache(self):
        self.calculated_workspace_size = {}
        self.algo_family = get_algo_family()
        self.max_workspace_size = 8 * 2 ** 28  # tune this parameter
        self.autotuner_cache = {{}, {}, {}}

    def forward_algorithm(self, link, params):
        alg_search_mode = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
        return self.setup_algo(link, alg_search_mode, params)

    def check_iteration(self, link, findAPI_idx):
        # TODO: work on this more
        if warmup_iterations == 0:
            return
    



    def setup_algo(self, link, findAPI_idx, alg_search_mode, params):
        # findAPI_idx:  0, 1, 2  (0: forward, 1: backward, 2:backward update)
        self.check_iteration(link, 0)

        cur_workspace, cur_workspace_size = 8 * 2**10, 8 * 2**10

        API = 


it = None
def get():
    # TODO: support multiple devices
    if it is None:
        it = Finder()
    return it
    
# amount of memory to use on 1st iteration for FindEx
initial_workspace_bytes = 1024



shared_buffer = {}
def shared_buf_for_stream(device, stream=0):
    if device not in shared_buffer:
        shared_buffer[device] = {}
    if not stream in shared_buffer[device]:
        buf = {'current_size': initial_workspace_bytes, 
               'next_size': -1}
        allocate_storage(buf)
        shared_buffer[device][stream] = buf
    return shared_buffer[device][stream]



def allocate_storage(buf):
    if buf['next_size'] < 0:
        buf['next_size'] = buf['current_size']

    el_size = 8.
    # get number of elements in the buf, roundup
    newelem = math.floor((buf['next_size'] + el_size - 1) / el_size)

    # TODO
    if 'storage' in buf:
        pass
    else:
        buf['storage'] = ...

    buf['current_size'] = ...
    buf['data'] = ...
    buf['next_size'] = -1


def get_shared_workspace():
    device = cupy.cuda.get_device_id()
    buf = shared_buf_for_stream(device)
    return buf['data'], buf['current_size']
    

def calculate_max_workspace_size(reserve=None, fraction=None):
    if reserve is None:
        reserve = 1024 * 1024
    if fraction is None:
        fraction = 95 / 100.

    buf, cur_size = get_shared_workspace()

    # TODO: add support for multiple devices
    free_memory, total_memory = cupy.cuda.runtime.memGetInfo()  
    new_size = (free_memory + cur_size - reserve) * fraction




class Convolution2D(link.Link):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None, deterministic=False):
        super(Convolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.use_cudnn = use_cudnn
        self.out_channels = out_channels
        self.deterministic = deterministic

        # For backward compatibility
        self.initialW = initialW
        self.wscale = wscale

        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        self._W_initializer = initializers._get_initializer(
            initialW, scale=math.sqrt(wscale))

        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initilizer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=bias_initilizer)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        self.add_param('W', W_shape, initializer=self._W_initializer)

    def create_IODescriptor(self, x):
        groups = 1  # fix this for now
        exists = False
        if exists:
            handle = cudnn.get_handle()
            # create input descriptor
            x_slice = x  # TODO: this changes when groups != 1
            x_desc = cudnn.create_tensor_descriptor(x_slice)

            # create conv descriptor
            conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx), x.dtype)

            # get output shape, resize output


            out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph,
                                        cover_all=self.cover_all)
            assert out_h > 0, 'Height in the output should be positive.'
            out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw,
                                        cover_all=self.cover_all)
            assert out_w > 0, 'Width in the output should be positive.'

            y = cuda.cupy.empty((n, out_channels, out_h, out_w), dtype=x.dtype)
            y_desc = cudnn.create_tensor_descriptor(y)

            # prepare

            # create offsets for groups (?)

    def prepare(self, x, y):
        def shape(x):
            return ','.join([str(elem) for elem in x.shape])

        def vals(x):
            return ','.join(x)

        self.autotuner_hash = '-dimA{} -filtA{} {} -padA{} -convStrideA{} {}'.format(
            shape(x), shape(self.W), shape(y), vals(self.pad), vals(self.stride), self.W.dtype)

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])

        self.create_IODescriptor(x)
        finder = get()
        finder

        return convolution_2d.convolution_2d(
            x, self.W, self.b, self.stride, self.pad, self.use_cudnn,
            deterministic=self.deterministic)

