import sys
import json
import shutil
from pathlib import Path
from omegaconf import OmegaConf


def parse_config():
    result = []
    for arg in sys.argv[1:]:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            conf = OmegaConf.to_container(OmegaConf.load(arg), resolve=True)
            mode = f"swift/cli/{conf.pop('stage')}.py"
            debug_mode = conf.pop('debug', False)
            for key, value in conf.items():
                if key == 'output_dir' and debug_mode:
                    value = str(Path(value).with_name('temp'))
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
    return mode, result


if __name__ == '__main__':
    mode, args = parse_config()
    print(f'<randy>{mode} {args}</randy>')
