# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys


def _use_ray() -> bool:
    if '--use_ray' not in sys.argv:
        return False
    idx = sys.argv.index('--use_ray')
    sys.argv.pop(idx)
    if idx < len(sys.argv) and sys.argv[idx].lower() in ('true', 'false'):
        val = sys.argv.pop(idx).lower() == 'true'
        return val
    return True


if __name__ == '__main__':
    if _use_ray():
        from swift.ray.megatron.pipeline import main as ray_main
        ray_main()
    else:
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
        from swift.megatron import megatron_rlhf_main
        megatron_rlhf_main()
