from __future__ import division
from collections import namedtuple
import multiprocessing
from multiprocessing import sharedctypes
import signal
import sys
import threading
import warnings

import copy
import numpy
import six

from chainer.dataset import iterator
from chainer.iterators.samplers.linear_sampler import LinearSampler
from chainer.iterators.samplers.sampler import get_state


_response_time = 1.
_short_time = 0.001
_PrefetchState = namedtuple('_PrefetchState', (
    'epoch', 'is_new_epoch', 'current_epoch_detail', 'previous_epoch_detail',
    'sampler_state'
))


class MultiprocessIterator(iterator.Iterator):

    """Dataset iterator that loads examples in parallel.

    This is an implementation of :class:`~chainer.dataset.Iterator` that loads
    examples with worker processes. It uses the standard :mod:`multiprocessing`
    module to parallelize the loading. The dataset is sent to the worker
    processes in the standard way using pickle.

    Note that this iterator effectively prefetches the examples for the next
    batch asynchronously after the current batch is returned.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.
        n_processes (int): Number of worker processes. The number of CPUs is
            used by default.
        n_prefetch (int): Number of prefetch batches.
        shared_mem (int): The size of using shared memory per data.
            If ``None``, size is adjusted automatically.

    """

    _interruption_testing = False  # for testing

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_processes=None, n_prefetch=1, shared_mem=None,
                 sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size

        self.n_processes = n_processes or multiprocessing.cpu_count()
        self.n_prefetch = max(n_prefetch, 1)
        self.shared_mem = shared_mem
        if sampler is None:
            sampler = LinearSampler(
                len(dataset), batch_size, shuffle, repeat)
        self.sampler = sampler
        self._sampler_state = get_state(self.sampler)

        self._finalized = False

        self._comm = _Communicator(self.n_prefetch)
        self.reset()

        self._prefetch_loop = _PrefetchLoop(
            self.dataset, self.batch_size,
            self.n_processes, self.n_prefetch, self.shared_mem, self._comm,
            self.sampler,
            self._interruption_testing)
        # defer launching prefetch thread until creating the worker pool,
        # not to leave a background thread in forked processes.
        self._thread = None

    def __next__(self):
        measure_mode = False
        if self._thread is None:
            if self._prefetch_loop.measure_required():
                measure_mode = True
                batch, prefetch_state = self._prefetch_loop.measure()
            self._thread = self._prefetch_loop.launch_thread()
            del self._prefetch_loop

        if not measure_mode:
            batch, prefetch_state = self._comm.get()

        (self.epoch, self.is_new_epoch, self._current_epoch_detail,
         self._previous_epoch_detail,
         self._sampler_state) = prefetch_state
        if batch is None:
            raise StopIteration
        else:
            return batch

    next = __next__

    def __del__(self):
        # When `self.__del__()` is called, `self.__init__()` may not be
        # finished. So some attributes may be undefined.
        if not hasattr(self, '_finalized'):
            # We don't know how to finalize this uninitialized object
            return
        if not hasattr(self, '_comm'):
            self._comm = None
        if not hasattr(self, '_thread'):
            self._thread = None

        if self._finalized:
            return
        self._finalized = True
        if self._comm is None:
            return
        self._comm.terminate()

        if self._thread is None:
            return
        while self._thread.is_alive():
            self._thread.join(_response_time)

    finalize = __del__

    def __copy__(self):
        sampler = copy.copy(self.sampler)
        other = MultiprocessIterator(
            self.dataset, self.batch_size,
            n_processes=self.n_processes, n_prefetch=self.n_prefetch,
            shared_mem=self.shared_mem,
            sampler=sampler)
        # I think it needs to copy other stuffs, like _comm

        other.epoch = self.epoch
        other.is_new_epoch = self.is_new_epoch
        other._current_epoch_detail = self._current_epoch_detail
        other._previous_epoch_detail = self._previous_epoch_detail

        other._set_prefetch_state()
        return other

    @property
    def epoch_detail(self):
        return self._current_epoch_detail

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        self._current_epoch_detail = serializer(
            'epoch_detail', self._current_epoch_detail)
        self._previous_epoch_detail = serializer(
            'previous_epoch_detail', self._previous_epoch_detail)

        for key, val in self._sampler_state.items():
            val = serializer('sampler_state' + '/' + key, val)
            setattr(self.sampler, key, val)
        self._set_prefetch_state()

    def reset(self):
        if self._finalized:
            raise NotImplementedError(
                'Reset of finalized MultiProcessIterator is currently not '
                'supported.')
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self._current_epoch_detail = 0.
        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
        self.sampler.reset()

        self._set_prefetch_state()

    def _set_prefetch_state(self):
        prefetch_state = _PrefetchState(
            epoch=self.epoch,
            is_new_epoch=self.is_new_epoch,
            current_epoch_detail=self._current_epoch_detail,
            previous_epoch_detail=self._previous_epoch_detail,
            sampler_state=get_state(self.sampler)
        )
        self._comm.reset(prefetch_state)


class _Communicator(object):

    STATUS_CONTINUE = 0
    STATUS_RESET = 1
    STATUS_TERMINATE = 2

    def __init__(self, n_prefetch):
        self.n_prefetch = n_prefetch

        self._lock = threading.Lock()
        self._not_empty_cond = threading.Condition(self._lock)
        self._not_full_cond = threading.Condition(self._lock)
        self._batch_queue = []
        self._status = _Communicator.STATUS_CONTINUE
        self._reset_count = 0

    @property
    def is_terminated(self):
        with self._lock:
            return self._status == _Communicator.STATUS_TERMINATE

    # called from iterator
    def get(self):
        with self._lock:
            while len(self._batch_queue) == 0:
                self._not_empty_cond.wait(_response_time)
            batch, prefetch_state = self._batch_queue.pop(0)
            self._not_full_cond.notify()
            return batch, prefetch_state

    # called from iterator
    def reset(self, prefetch_state):
        with self._lock:
            self._status = _Communicator.STATUS_RESET
            self._prefetch_state = prefetch_state
            self._batch_queue = []
            self._not_full_cond.notify()
            self._reset_count += 1

    # called from iterator
    def terminate(self):
        with self._lock:
            self._status = _Communicator.STATUS_TERMINATE
            self._batch_queue = []
            self._not_full_cond.notify()
            self._reset_count += 1

    # called from thread
    def check(self):
        with self._lock:
            status = self._status
            self._status = _Communicator.STATUS_CONTINUE
            prefetch_state = None
            if status == _Communicator.STATUS_RESET:
                prefetch_state = self._prefetch_state
            return status, prefetch_state, self._reset_count

    # called from thread
    def put(self, batch, prefetch_state, reset_count):
        with self._lock:
            if len(self._batch_queue) == self.n_prefetch:
                self._not_full_cond.wait()
            if reset_count == self._reset_count:
                self._batch_queue.append((batch, prefetch_state))
                self._not_empty_cond.notify()


class _PrefetchLoop(object):

    def __init__(self, dataset, batch_size,
                 n_processes, n_prefetch, mem_size, comm,
                 sampler,
                 _interruption_testing):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.mem_size = mem_size
        self.comm = comm
        self.sampler = sampler

        self._allocate_shared_memory()
        self._pool = None

        self._interruption_testing = _interruption_testing

    def measure_required(self):
        return self.mem_size is None

    def measure(self):
        status, prefetch_state, _ = self.comm.check()
        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state

        indices = self._proceed()
        if indices is None:  # stop iteration
            batch = None
        else:
            batch = [self.dataset[idx] for idx in indices]
            self.mem_size = max(map(_measure, batch))
            self._allocate_shared_memory()

        return batch, self.prefetch_state

    def _allocate_shared_memory(self):
        if self.measure_required():
            self.mem_bulk = None
        else:
            self.mem_bulk = \
                sharedctypes.RawArray('b', self.batch_size * self.mem_size)

    def launch_thread(self):
        self._pool = multiprocessing.Pool(
            processes=self.n_processes,
            initializer=_fetch_setup,
            initargs=(self.dataset, self.mem_size, self.mem_bulk))
        if self._interruption_testing:
            pids = self._pool.map(_report_pid, range(self.n_processes))
            print(' '.join(map(str, pids)))
            sys.stdout.flush()

        thread = threading.Thread(target=self._run, name='prefetch_loop')
        thread.setDaemon(True)
        thread.start()
        return thread

    def _run(self):
        alive = True
        try:
            while alive:
                alive = self._task()
        finally:
            self._pool.close()
            self._pool.join()

    def _task(self):
        status, prefetch_state, reset_count = self.comm.check()
        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state
        elif status == _Communicator.STATUS_TERMINATE:
            return False  # stop loop

        indices = self._proceed()
        if indices is None:  # stop iteration
            batch = None
        else:
            future = self._pool.map_async(_fetch_run, enumerate(indices))
            while True:
                try:
                    data_all = future.get(_response_time)
                except multiprocessing.TimeoutError:
                    if self.comm.is_terminated:
                        return False
                else:
                    break

            batch = [_unpack(data, self.mem_bulk) for data in data_all]

        self.comm.put(batch, self.prefetch_state, reset_count)
        return True

    def _proceed(self):
        (epoch, _, previous_epoch_detail, _, previous_state) = self.prefetch_state

        try:
            indices = self.sampler.get_indices(previous_state)
        except StopIteration:
            return None

        if len(indices) > self.batch_size:
            raise ValueError(
                'The number of indices sampled from a sampler should '
                'be less than or equal to batch_size.')

        if self.sampler.is_new_epoch:
            epoch += 1
        
        current_epoch_detail = epoch + self.sampler.epoch_percentage
        state = get_state(self.sampler)
        self.prefetch_state = _PrefetchState(
            epoch, self.sampler.is_new_epoch, current_epoch_detail,
            previous_epoch_detail, state)
        return indices


# Using `parametarized` funciton (e.g. bound method) with Pool is tricky due to
# restrictions imposed by Pickle. Picklable types differ across versions.
# Just using top-level function with globals seems to be safest.
# it doesn't mean thread safety broken or global variables visible;
# notice that each process uses different address space.
# To make static linter happy, we first initialize global variables.
_fetch_dataset = None
_fetch_mem_size = None
_fetch_mem_bulk = None


def _fetch_setup(dataset, mem_size, mem_bulk):
    global _fetch_dataset, _fetch_mem_size, _fetch_mem_bulk
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _fetch_dataset = dataset
    _fetch_mem_size = mem_size
    _fetch_mem_bulk = mem_bulk


def _fetch_run(inputs):
    i, index = inputs
    data = _fetch_dataset[index]
    if _fetch_mem_bulk is not None:
        offset = i * _fetch_mem_size
        limit = offset + _fetch_mem_size
        data = _pack(data, _fetch_mem_bulk, offset, limit)
    return data


def _report_pid(_):  # for testing
    return multiprocessing.current_process().pid


class _PackedNdarray(object):

    def __init__(self, array, mem, offset):
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.size = array.size
        self.offset = offset
        total = self.offset + self.nbytes
        if total > len(mem):
            raise ValueError(
                'Shared memory size is too small. expect:{}, actual:{}'.format(
                    total, len(mem)))
        target = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        target[...] = array.ravel()

    def unpack(self, mem):
        ret = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        ret = ret.reshape(self.shape).copy()
        return ret


def _measure(data):
    expect = 0
    t = type(data)
    if t is tuple or t is list or t is dict:
        for v in data:
            if isinstance(v, numpy.ndarray):
                expect += v.nbytes
    return expect


def _pack(data, mem, offset, limit):
    if len(mem) == 0:
        return data
    t = type(data)
    over = False
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > limit:
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > limit:
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret[k] = v
        data = ret
    if over:
        expect = _measure(data)
        warnings.warn(
            'Shared memory size is too small.\n' +
            'Please set shared_mem option for MultiprocessIterator.\n' +
            'Expect shared memory size: {} bytes.\n'.format(expect) +
            'Actual shared memory size: {} bytes.'.format(limit - offset),
            UserWarning)
    return data


def _unpack(data, mem):
    if len(mem) == 0:
        return data
    t = type(data)
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret[k] = v
        data = ret
    return data
