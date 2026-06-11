import sys
import json
import shutil
from pathlib import Path
from omegaconf import OmegaConf


def resolve_ckpt(path):
    path = Path(path)
    last = lambda p: (
        sorted(
            p.glob('checkpoint-*'),
            key=lambda x: int(x.name.split('-')[-1])
        ) or [p]
    )[-1]
    if path.exists():
        return path if path.name.startswith('checkpoint-') else last(path)
    return last(sorted(path.parent.glob(f'{path.name}-*'))[-1])


def parse_config():
    result = []
    for arg in sys.argv[1:]:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            conf = OmegaConf.to_container(OmegaConf.load(arg), resolve=True)
            debug_mode = conf.pop('debug', False)
            is_megatron = conf.pop('megatron', False)
            stage = '_megatron/' if is_megatron else ''
            entry = f"swift/cli/{stage}{conf.pop('stage')}.py"
            for key, value in conf.items():
                if key == 'model':
                    value = resolve_ckpt(value)
                if key == 'output_dir' and debug_mode:
                    value = Path(value).with_name('temp')
                    shutil.rmtree(value, ignore_errors=True)
                result.append(f'--{key}')
                if isinstance(value, dict):
                    result.append(f"'{json.dumps(value, ensure_ascii=False)}'")
                else:
                    if not isinstance(value, list):
                        value = str(value).split(',')
                    result.extend(list(map(str, value)))
        else:
            result.append(arg)
    result = ' '.join(result)
    return entry, result


if __name__ == '__main__':
    entry, args = parse_config()
    print(f'<randy>{entry} {args}</randy>')
