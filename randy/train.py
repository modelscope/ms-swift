import sys
import json
from omegaconf import OmegaConf


def parse_args():
    result = []
    for arg in sys.argv[1:]:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            conf = parse_config(arg)
            mode = conf.pop('stage')
            path = conf.pop('dataset_info')
            info = OmegaConf.load(path)
            info = OmegaConf.to_container(info, resolve=True)
            for key, value in conf.items():
                result.append(f'--{key}')
                if key in ['dataset', 'val_dataset', 'eval_dataset']:
                    names = value.split(',')
                    for i, name in enumerate(names):
                        try:
                            name, *_ = name.split('#', 1)
                            path = info[name]['file_name']
                            names[i] = '#'.join([path] + _)
                        except:
                            pass
                    result.extend(names)
                elif isinstance(value, list):
                    result.extend(value)
                else:
                    result.append(value)
        else:
            result.append(arg)
    result = ' '.join(result)
    return mode, result


def parse_config(path):
    conf = OmegaConf.load(path)
    conf = OmegaConf.to_container(conf, resolve=True)
    for key, value in conf.items():
        if isinstance(value, dict):
            conf[key] = f"'{json.dumps(value, ensure_ascii=False)}'"
        elif isinstance(value, list):
            conf[key] = list(map(str, value))
        else:
            conf[key] = str(value)
    return conf


if __name__ == '__main__':
    mode, args = parse_args()
    entrypoint = f'swift/cli/{mode}.py'
    print(f'<randy>{entrypoint} {args}</randy>')
