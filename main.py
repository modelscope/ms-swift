from omegaconf import OmegaConf
from importlib import import_module
from swift.llm import TrainArguments


def parse_config(path):
    conf = OmegaConf.load(path)
    conf = OmegaConf.to_container(conf, resolve=True)
    mode = conf.pop('stage')
    path = conf.pop('dataset_info')
    info = OmegaConf.load(path)
    info = OmegaConf.to_container(info, resolve=True)
    for key, value in conf.items():
        if key in ['dataset', 'val_dataset', 'eval_dataset']:
            names = value.split(',')
            for i, name in enumerate(names):
                try:
                    name, *_ = name.split('#', 1)
                    path = info[name]['file_name']
                    names[i] = '#'.join([path] + _)
                except:
                    pass
            conf[key] = names
    conf.pop('deepspeed', None)
    conf.pop('use_liger_kernel', None)
    return mode, conf


if __name__ == '__main__':
    mode, conf = parse_config('randy/demo.yaml')
    entrypoint = getattr(import_module('swift.llm'), f'{mode}_main')
    entrypoint(TrainArguments(**conf))
