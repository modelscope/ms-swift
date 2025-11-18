import sys
from omegaconf import OmegaConf


def parse_args(argv):
    mode, args = '', []

    for arg in argv:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            conf = OmegaConf.load(arg)
            path = conf.pop('dataset_info')
            info = OmegaConf.load(path)
            mode = conf.pop('stage')

            for key, value in conf.items():
                if key in ['dataset', 'val_dataset']:
                    names = value.split(',')
                    for i, name in enumerate(names):
                        name, *rest = name.split('#', 1)
                        path = info[name]['file_name']
                        names[i] = '#'.join([path] + rest)
                    args.append(f'--{key}')
                    args.extend(names)
                else:
                    args.append(f'--{key}={value}')
        else:
            args.append(arg)

    return mode, args


if __name__ == '__main__':
    mode, args = parse_args(sys.argv[1:])
    arguments = ' '.join(args)
    entrypoint = f'swift/cli/{mode}.py'
    print(f'<randy>{entrypoint} {arguments}</randy>')
