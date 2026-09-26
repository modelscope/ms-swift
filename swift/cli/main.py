# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Optional

import json
import yaml

from swift.utils import get_logger

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'swift.cli.pt',
    'sft': 'swift.cli.sft',
    'infer': 'swift.cli.infer',
    'merge-lora': 'swift.cli.merge_lora',
    'web-ui': 'swift.cli.web_ui',
    'deploy': 'swift.cli.deploy',
    'rollout': 'swift.cli.rollout',
    'rlhf': 'swift.cli.rlhf',
    'sample': 'swift.cli.sample',
    'export': 'swift.cli.export',
    'eval': 'swift.cli.eval',
    'app': 'swift.cli.app',
}

DEV_ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'swift.dev.cli.pt',
    'sft': 'swift.dev.cli.sft',
    'rlhf': 'swift.dev.cli.rlhf',
    'infer': 'swift.dev.cli.infer',
    'merge-lora': 'swift.dev.cli.merge_lora',
    'deploy': 'swift.dev.cli.deploy',
    'export': 'swift.dev.cli.export',
    'eval': 'swift.dev.cli.eval',
}
DEV_MEGATRON_ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'swift.dev.cli.megatron_pt',
    'sft': 'swift.dev.cli.megatron',
    'rlhf': 'swift.dev.cli.megatron_rlhf',
    'export': 'swift.dev.cli.megatron_export',
}


def _use_dev_cli() -> bool:
    return os.environ.get('USE_SWIFT_V5', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def resolve_route(method_name: str, route_mapping: Dict[str, str], is_megatron: bool = False) -> str:
    if not _use_dev_cli():
        return route_mapping[method_name]
    if not is_megatron and method_name == 'web-ui':
        return route_mapping[method_name]
    dev_routes = DEV_MEGATRON_ROUTE_MAPPING if is_megatron else DEV_ROUTE_MAPPING
    if method_name not in dev_routes:
        raise RuntimeError(f'USE_SWIFT_V5 is enabled but no dev route exists for {method_name!r}.')
    return dev_routes[method_name]


def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def parse_yaml_args(argv):  # noqa: C901
    if not argv:
        return
    config = None
    if argv[0].endswith('.json'):
        with open(argv[0], 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif argv[0].endswith('.yaml') or argv[0].endswith('.yml'):
        with open(argv[0], 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    if config is None:
        return
    # Used for saving configurations
    os.environ['SWIFT_CONFIG_FILE'] = argv[0]

    env = config.pop('ENV', None)
    if env:
        for k, v in env.items():
            if k not in os.environ:
                os.environ[k] = str(v)
            elif str(v) != os.environ[k]:
                logger.warning(f'{k} is already set in environment, using `{os.environ[k]}` instead of `{v}`')
    config_argv = []
    for k, v in config.items():
        config_argv.append(f'--{k}')
        if isinstance(v, list):
            config_argv += v
        else:
            if isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            else:
                v = str(v)
            config_argv.append(v)
    argv[0:1] = config_argv


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def cli_main(route_mapping: Optional[Dict[str, str]] = None, is_megatron: bool = False) -> None:
    route_mapping = dict(route_mapping or ROUTE_MAPPING)
    argv = sys.argv[1:]
    method_name = argv[0].replace('_', '-')
    argv = argv[1:]
    route = resolve_route(method_name, route_mapping, is_megatron)
    is_dev_route = route.startswith('swift.dev.')
    file_path = None if is_dev_route else importlib.util.find_spec(route).origin
    parse_yaml_args(argv)
    torchrun_args = get_torchrun_args()
    python_cmd = sys.executable
    if torchrun_args is None or (not is_megatron and method_name not in {'pt', 'sft', 'rlhf', 'infer'}):
        args = [python_cmd, '-m', route, *argv] if is_dev_route else [python_cmd, file_path, *argv]
    elif is_dev_route:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, '--module', route, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    cli_main()
