# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import os
import subprocess
import sys
import json
from typing import Dict, List, Optional, Any

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


def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


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


def prepare_config_args(argv):
    for i in range(0, len(argv[1:]), 2):
        arg_name = argv[i]
        arg_value = argv[i + 1]
        if arg_name == '--config':
            from omegaconf import OmegaConf, DictConfig
            from swift.ray import RayHelper
            config = OmegaConf.load(arg_value)

            def parse_dict_config(cfg: DictConfig) -> Dict[str, Any]:
                result = {}
                def _traverse(config: Any, parent_key: str = ""):
                    if isinstance(config, DictConfig):
                        for key, value in config.items():
                            if key == 'device_groups':
                                result[key] = json.dumps(OmegaConf.to_container(value))
                            else:
                                current_path = f"{parent_key}.{key}" if parent_key else key
                                _traverse(value, current_path)
                    else:
                        last_key = parent_key.split('.')[-1] if parent_key else ""
                        result[last_key] = config

                _traverse(cfg)
                return result
                
            cfg = parse_dict_config(config)
            for key, value in cfg.items():
                argv.append(f'--{key}')
                if not isinstance(value, str):
                    value = str(value)
                argv.append(value)
            
            argv.pop(i)
            argv.pop(i)
            break


def _compat_web_ui(argv):
    # [compat]
    method_name = argv[0]
    if method_name in {'web-ui', 'web_ui'} and ('--model' in argv or '--adapters' in argv or '--ckpt_dir' in argv):
        argv[0] = 'app'
        logger.warning('Please use `swift app`.')


def cli_main(route_mapping: Optional[Dict[str, str]] = None) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    _compat_web_ui(argv)
    method_name = argv[0].replace('_', '-')
    argv = argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    torchrun_args = get_torchrun_args()
    prepare_config_args(argv)
    python_cmd = sys.executable
    if torchrun_args is None or method_name not in {'pt', 'sft', 'rlhf', 'infer'}:
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    cli_main()
