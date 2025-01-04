

class Sampler:

    def __call__(self, model, batch, generation_config):
        raise NotImplementedError


samplers = {}
