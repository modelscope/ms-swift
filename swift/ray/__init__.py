# Copyright (c) Alibaba, Inc. and its affiliates.
from .base import RayHelper


def try_init_ray():
    import json
    import argparse
    from transformers.utils import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ray', type=str, default='0')
    parser.add_argument('--device_groups', type=str, default=None)
    args, _ = parser.parse_known_args()
    args.use_ray = strtobool(args.use_ray)
    if args.use_ray:
        RayHelper.initialize(json.loads(args.device_groups))
