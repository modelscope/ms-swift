import math
from typing import Dict, Any

import transformers
from packaging import version
from swift import get_logger

from swift.llm.utils.utils import set_rope_scaling, get_rope_scaling

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
    max_position_embeddings = get_max_model_len(model_config, ignore_rope_scaling=True)
    if rope_scaling and max_position_embeddings:
        max_length = max_length or max_position_embeddings
        rope_scaling_factor = max(float(math.ceil(max_length / max_position_embeddings)), 1.0)
        set_rope_scaling(model_config, {'type': rope_scaling, 'factor': rope_scaling_factor})
        logger.info(f'rope_scaling is set to type: {get_rope_scaling(model_config)}')


def patch_tokenizer(tokenizer, eos_token, pad_token, placeholder_tokens):
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
    # multimodal
    llm_config = None
    for k in ['language_config', 'llm_config', 'text_config']:
        llm_config = getattr(model_config, k, None)
        if llm_config:
            break
    if llm_config and hasattr(llm_config, 'hidden_size') and not hasattr(model_config, 'hidden_size'):
        model_config.hidden_size = llm_config.hidden_size
