from __future__ import division
import numpy

from chainer.iterators.samplers.sampler import Sampler


class LinearSampler(Sampler):

    state_attributes = ['pos', 'order']

    def __init__(self, length, batch_size, shuffle, repeat):
        self.length = length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pos = 0

        if self.shuffle:
            self.order = numpy.random.permutation(self.length)
        else:
            self.order = None
        self.repeat = repeat

        self._is_new_epoch = False

    @property
    def epoch_percentage(self):
        return self.pos / self.length

    @property
    def is_new_epoch(self):
        return self._is_new_epoch

    def get_indices(self, state):
        pos = state['pos']
        order = state['order']
        if self.is_new_epoch and not self.repeat:
            raise StopIteration

        new_pos = pos + self.batch_size
        if new_pos < self.length:
            self._is_new_epoch = False
            if order is None:
                indices = numpy.arange(pos, new_pos)
            else:
                indices = order[pos:new_pos]
        else:
            new_pos = new_pos - self.length if self.repeat else 0
            self._is_new_epoch = True
            if order is None:
                indices = numpy.arange(pos, self.length)
                if self.repeat:
                    indices = \
                        numpy.concatenate((indices, numpy.arange(new_pos)))
            else:
                indices = order[pos:self.length]
                if self.repeat:
                    order = numpy.random.permutation(self.length)
                    indices = \
                        numpy.concatenate((indices, order[:new_pos]))
        self.pos = new_pos
        self.order = order
        return indices

    def reset(self):
        self.pos = 0
        self._is_new_epoch = False
        if self.shuffle:
            self.order = numpy.random.permutation(self.length)
        else:
            self.order = None
