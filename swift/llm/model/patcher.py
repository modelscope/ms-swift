# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from accelerate.utils import find_device

from swift.llm import to_device
from swift.utils import get_logger

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
        return output.requires_grad_(True).clone()

    module.register_forward_hook(_clone_hook)


def patch_output_to_input_device(module: torch.nn.Module):
    """Patch the module, to make sure the output is in the same device with the input.

    Args:
        module: The module to be patched
    """

    def _output_to_input_device_hook(module, args, kwargs, output):
        device = find_device(args) or find_device(kwargs)
        return to_device(output, device)

    module.register_forward_hook(_output_to_input_device_hook, with_kwargs=True)
