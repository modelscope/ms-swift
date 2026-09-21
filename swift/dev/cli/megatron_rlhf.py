"""Megatron RLHF CLI over the shared dev RLHF Config surface."""
from __future__ import annotations
import os
from typing import List, Optional


def megatron_rlhf_main(argv: Optional[List[str]] = None):
    from swift.dev.cli.rlhf import parse_rlhf_configs, run_rlhf_configs

    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    return run_rlhf_configs(parse_rlhf_configs(argv, megatron=True))


if __name__ == '__main__':
    megatron_rlhf_main()
