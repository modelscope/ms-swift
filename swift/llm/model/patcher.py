import math
from typing import Dict, Any

import torch
import transformers
from packaging import version
from swift import get_logger

from swift.llm.utils.utils import set_rope_scaling, get_rope_scaling, to_device

from swift.llm import get_max_model_len
from transformers import GPTQConfig

logger = get_logger()


def patch_gptq_model(bits: int, model_config, model_kwargs: Dict[str, Any]) -> None:
    """Patch autogptq model:
        1. Fix the no grad problem in training
        2. Fix the slow generation speed in inference

    Args:
        bits: The quantized bit
        model_config: The model config
        model_kwargs: The model kwargs
    """
    assert model_kwargs.get('quantization_config') is None
    if bits == 0:
        bits = model_config.quantization_config['bits']
    if version.parse(transformers.__version__) >= version.parse('4.35'):
        model_kwargs['quantization_config'] = GPTQConfig(bits=bits, use_exllama=False)
    else:
        model_kwargs['quantization_config'] = GPTQConfig(bits=bits, disable_exllama=True)

    # fix quantlinear bug
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
    __old_forward = QuantLinear.forward

    def _new_forward(self, x):
        if not self.training or not self.autogptq_cuda_available:
            return self.__old_forward(x)
        # fix sft no grad
        self.autogptq_cuda_available = False
        res = self.__old_forward(x)
        self.autogptq_cuda_available = True
        return res

    if not hasattr(QuantLinear, '__old_forward'):  # avoid double patching
        QuantLinear.__old_forward = __old_forward
        QuantLinear.forward = _new_forward


def patch_rope_scaling(model_config, rope_scaling, max_length):
    """Patch rope scaling, to enable dynamic/linear rope

    Args:
        model_config:
        rope_scaling:
        max_length:

    Returns:

    """
    max_position_embeddings = get_max_model_len(model_config, ignore_rope_scaling=True)
    if rope_scaling and max_position_embeddings:
        max_length = max_length or max_position_embeddings
        rope_scaling_factor = max(float(math.ceil(max_length / max_position_embeddings)), 1.0)
        set_rope_scaling(model_config, {'type': rope_scaling, 'factor': rope_scaling_factor})
        logger.info(f'rope_scaling is set to type: {get_rope_scaling(model_config)}')


def patch_tokenizer(tokenizer, eos_token, pad_token, placeholder_tokens):
    """Patch tokenizer to add extra eos_token/pad_token/placeholder_tokens.

    Args:
        tokenizer: The tokenizer to be patched
        eos_token: The eos_token
        pad_token: The pad_token
        placeholder_tokens: The placeholder_tokens
    """
    if isinstance(eos_token, str):
        tokenizer.eos_token = eos_token
    elif isinstance(eos_token, int):
        tokenizer.eos_token_id = eos_token
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    if placeholder_tokens is not None:
        tokenizer.placeholder_tokens = placeholder_tokens
        tokenizer.placeholder_tokens_id = [tokenizer.convert_tokens_to_ids(token) for token in placeholder_tokens]


def patch_hidden_size(model_config):
    """Sometimes model config need `hidden_size` key, this will copy the value in llm domain to the outer config.

    Args:
        model_config: The model config
    """
    # multimodal
    llm_config = None
    for k in ['language_config', 'llm_config', 'text_config']:
        llm_config = getattr(model_config, k, None)
        if llm_config:
            break
    if llm_config and hasattr(llm_config, 'hidden_size') and not hasattr(model_config, 'hidden_size'):
        model_config.hidden_size = llm_config.hidden_size


def patch_device(model: torch.nn.Module):

    def get_device_hook(device):
        def _device_hook(module, input, output):
            return to_device(output, device)

        return _device_hook





def patch_output_clone(module: torch.nn.Module):

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

    def _output_device_map_hook(module, input, output: torch.Tensor):
        return output.to(input[0].device)

    module.register_forward_hook(_output_device_map_hook)
