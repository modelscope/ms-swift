from omegaconf import OmegaConf
from swift import sft_main, SftArguments


def parse_config(path):
    conf = OmegaConf.load(path)
    conf = OmegaConf.to_container(conf, resolve=True)
    for key, value in conf.items():
        if isinstance(value, str) and ',' in value:
            conf[key] = value.split(',')
    conf.pop('stage', None)
    conf.pop('deepspeed', None)
    conf.pop('use_liger_kernel', None)
    return conf


if __name__ == '__main__':
    conf = parse_config('randy/demo.yaml')
    sft_main(SftArguments(**conf))
