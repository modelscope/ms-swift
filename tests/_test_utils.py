import os
import torch


def is_npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except ImportError:
        return False


def setup_device_env(device_ids='0'):
    if is_npu_available():
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device_ids
    elif torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
