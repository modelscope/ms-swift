# Copyright (c) ModelScope Contributors. All rights reserved.
"""Legacy Ray integration for the ``transformers`` training backend.

This module hosts the original (pre-``swift.ray.megatron``) Ray
helpers — ``RayHelper`` / ``RayArguments`` / ``try_init_ray``.  They
are the only Ray entry point supported for the ``transformers`` /
``trainer_pytorch`` training path; ``device_groups`` + single-driver
``WorkerGroup`` plumbing lives there.

For Megatron-based RLHF (GRPO / GKD / DPO) use
``swift.ray.megatron.pipeline`` instead — that module is free-standing
and does NOT import from here.
"""


def __getattr__(name):
    if name == 'RayArguments':
        from .arguments import RayArguments
        return RayArguments
    if name == 'RayHelper':
        from .base import RayHelper
        return RayHelper
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def try_init_ray():
    import argparse
    import json
    from transformers.utils import strtobool

    from .base import RayHelper
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ray', type=str, default='0')
    parser.add_argument('--device_groups', type=str, default=None)
    args, _ = parser.parse_known_args()
    args.use_ray = strtobool(args.use_ray)
    if args.use_ray:
        RayHelper.initialize(json.loads(args.device_groups))
