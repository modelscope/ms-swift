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
    'merge': 'swift.dev.cli.merge',
    'deploy': 'swift.dev.cli.deploy',
    'export': 'swift.dev.cli.export',
    'eval': 'swift.dev.cli.eval',
}


def _use_dev_cli() -> bool:
    return os.environ.get('USE_SWIFT_V5', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def resolve_route(method_name: str, route_mapping: Dict[str, str], is_megatron: bool = False) -> str:
    if not _use_dev_cli():
        return route_mapping[method_name]
    if is_megatron:
        # twinkle unifies the Megatron and Transformers stacks, so v5 has no separate `megatron`
        # command: the backend is a single flag on the shared command. Redirect instead of routing.
        raise RuntimeError('USE_SWIFT_V5 is enabled, but v5 has no separate `megatron` command. '
                           f'Use `swift {method_name} --backend megatron` instead.')
    if method_name == 'web-ui':
        return route_mapping[method_name]
    if method_name not in DEV_ROUTE_MAPPING:
        raise RuntimeError(f'USE_SWIFT_V5 is enabled but no dev route exists for {method_name!r}.')
    return DEV_ROUTE_MAPPING[method_name]


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
    distributed_methods = {'pt', 'sft', 'rlhf', 'infer'}
    if is_dev_route:
        # `swift export --backend megatron` (mcore convert) is distributed as well, so the dev export
        # entry also relaunches under torchrun when NPROC_PER_NODE/NNODES is set.
        distributed_methods = distributed_methods | {'export'}
    if torchrun_args is None or (not is_megatron and method_name not in distributed_methods):
        args = [python_cmd, '-m', route, *argv] if is_dev_route else [python_cmd, file_path, *argv]
    elif is_dev_route:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, '--module', route, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    # Not subprocess.run: its `except: process.kill()` sends SIGKILL to the child on KeyboardInterrupt.
    # Ctrl+C is delivered to the whole foreground process group, so the child (e.g. `swift infer`) has
    # already received SIGINT and is running its own graceful shutdown -- releasing the model, spinning
    # down the vLLM engine and its worker subprocesses. SIGKILLing it mid-shutdown is exactly what dumped
    # the parent traceback and leaked the workers' IPC semaphores/shared memory to resource_tracker. Use
    # Popen and keep waiting instead, so the child finishes cleanly and we exit with its real status.
    process = subprocess.Popen(args)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        try:
            # First Ctrl+C: let the child's graceful shutdown run to completion.
            returncode = process.wait()
        except KeyboardInterrupt:
            # Second Ctrl+C: the user really wants out now, so stop the child and stop waiting on it.
            process.terminate()
            returncode = process.wait()
    if returncode != 0:
        sys.exit(returncode)


if __name__ == '__main__':
    cli_main()
