# Copyright (c) ModelScope Contributors. All rights reserved.

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
