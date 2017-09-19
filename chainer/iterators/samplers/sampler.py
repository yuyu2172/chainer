def get_state(sampler):
    return {key: getattr(sampler, key)
            for key in sampler.state_attributes}


class Sampler(object):

    state_attributes = []

    @property
    def epoch_percentage(self):
        raise NotImplementedError

    @property
    def is_new_epoch(self):
        raise NotImplementedError

    def get_indices(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
