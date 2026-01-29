import sys
import json
from omegaconf import OmegaConf


def parse_config():
    result = []
    for arg in sys.argv[1:]:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            conf = OmegaConf.load(arg)
            conf = OmegaConf.to_container(conf, resolve=True)
            mode = conf.pop('stage')
            for key, value in conf.items():
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
