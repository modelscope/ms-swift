"""Megatron export CLI over the shared dev export pipeline."""
from __future__ import annotations
import os
from typing import List, Optional


def megatron_export_main(argv: Optional[List[str]] = None):
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    configs = parse_export_configs(argv, command='megatron_export')
    configs['distributed_config'].backend = 'megatron'
    configs['distributed_config'].nproc_per_node = configs['distributed_config'].nproc_per_node or int(
        os.environ.get('WORLD_SIZE', '1'))
    return run_export_configs(configs)


if __name__ == '__main__':
    megatron_export_main()
