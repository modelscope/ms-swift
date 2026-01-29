from omegaconf import OmegaConf
from importlib import import_module
from swift.llm import TrainArguments


def parse_config(path):
    conf = OmegaConf.load(path)
    conf = OmegaConf.to_container(conf, resolve=True)
    for key, value in conf.items():
        if isinstance(value, str) and ',' in value:
            conf[key] = value.split(',')
    mode = conf.pop('stage')
    conf.pop('deepspeed', None)
    conf.pop('use_liger_kernel', None)
    return mode, conf


if __name__ == '__main__':
    mode, conf = parse_config('randy/demo.yaml')
    entrypoint = getattr(import_module('swift.llm'), f'{mode}_main')
    entrypoint(TrainArguments(**conf))
