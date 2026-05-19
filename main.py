import shutil
from pathlib import Path
from omegaconf import OmegaConf
from swift import sft_main, SftArguments, export_main, ExportArguments


def parse_config(path):
    conf = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    debug_mode = conf.pop('debug', False)
    for key, value in conf.items():
        if key == 'output_dir' and debug_mode:
            value = str(Path(value).with_name('temp'))
            shutil.rmtree(value, ignore_errors=True)
        if isinstance(value, str) and ',' in value:
            conf[key] = value.split(',')
    conf.pop('stage', None)
    conf.pop('deepspeed', None)
    conf.pop('use_liger_kernel', None)
    return conf


if __name__ == '__main__':
    # conf = parse_config('randy/demo.yaml')
    # sft_main(SftArguments(**conf))

    conf = parse_config('randy/cache_data.yaml')
    export_main(ExportArguments(**conf))
