import math
from typing import Any, Dict, Mapping, Optional

import torch
from accelerate.utils import find_device

from swift import get_logger
from swift.llm import to_device

logger = get_logger()


def patch_fixed_device(module: torch.nn.Module, device):
    """Move the output to the specific device"""

    def get_device_hook(device):

        def _device_hook(module, input, output):
            return to_device(output, device)

        return _device_hook

    module.register_forward_hook(get_device_hook(device))


def patch_output_clone(module: torch.nn.Module):
    """Clone the output, to avoid the inplace problem"""

    def _clone_hook(module, input, output):
        if module.training:
            return output.requires_grad_(True).clone()
        else:
            return output

    module.register_forward_hook(_clone_hook)


def patch_output_to_input_device(module: torch.nn.Module):
    """Patch the module, to make sure the output is in the same device with the input.

    Args:
        module: The module to be patched
    """

    def recursive_set_device(data, device):
        if isinstance(data, Mapping):
            return type(data)({k: recursive_set_device(v, device) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(recursive_set_device(v, device) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {'device': device}
            return data.to(**kwargs)

    def _output_to_input_device_hook(module, args, kwargs, output):
        device = find_device(args) or find_device(kwargs)
        recursive_set_device(output, device)

    module.register_forward_hook(_output_to_input_device_hook, with_kwargs=True)
