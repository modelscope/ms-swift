# Copyright (c) ModelScope Contributors. All rights reserved.
import os


def sync_musa_visible_devices():
    """Honor CUDA_VISIBLE_DEVICES on Moore Threads GPUs.

    torch_musa only reads MUSA_VISIBLE_DEVICES, and torchada drops CUDA_VISIBLE_DEVICES when it is unset, so copy it
    over. Must run before torch is imported: torch autoloads torch_musa, which reads the variable only once.
    """
    if 'MUSA_VISIBLE_DEVICES' not in os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['MUSA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']


def try_use_single_device_mode():
    if os.environ.get('SWIFT_SINGLE_DEVICE_MODE', '0') == '1':
        # On MUSA the runtime reads MUSA_VISIBLE_DEVICES (see `sync_musa_visible_devices`).
        env_key = 'MUSA_VISIBLE_DEVICES' if 'MUSA_VISIBLE_DEVICES' in os.environ else 'CUDA_VISIBLE_DEVICES'
        visible_devices = os.environ.get(env_key)
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is None or not visible_devices:
            return
        visible_devices = visible_devices.split(',')
        visible_device = visible_devices[int(local_rank)]
        os.environ[env_key] = str(visible_device)
        os.environ['LOCAL_RANK'] = '0'
