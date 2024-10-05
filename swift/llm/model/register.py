import inspect
import os
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig,
                          PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)
from transformers.utils import is_torch_bf16_gpu_available
from transformers.utils.versions import require_version

from swift.llm import TemplateType
from swift.utils import get_logger, use_torchacc
from .utils import HfConfigFactory, safe_snapshot_download

MODEL_MAPPING: Dict[str, Dict[str, Any]] = {}

ARCH_MAPPING: Optional[Dict[str, Dict[str, List[str]]]] = None

GetModelTokenizerFunction = Callable[..., Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]]
logger = get_logger()


class Model(NamedTuple):
    ms_model_id: Optional[str] = None
    hf_model_id: Optional[str] = None
    model_path: Optional[str] = None
    ms_revision: Optional[str] = None


class TemplateGroup(NamedTuple):
    chat_template: str
    generation_template: Optional[str] = TemplateType.default_generation


class ModelGroup(NamedTuple):
    models: List[Model]
    tags: Optional[List[str]] = None


# [TODO:eos_token -> template]
def register_model(model_type: str,
                   architectures: str,
                   model_groups: Union[ModelGroup, List[ModelGroup]],
                   template: TemplateGroup,
                   get_function: GetModelTokenizerFunction,
                   *,
                   requires: Optional[List[str]] = None,
                   ignore_file_pattern: Optional[List[str]] = None,
                   support_flash_attn: bool = False,
                   support_vllm: bool = False,
                   support_lmdeploy: bool = False,
                   is_multimodal: bool = False,
                   is_moe: bool = False,
                   function_kwargs: Optional[Dict[str, Any]] = None,
                   exist_ok: bool = False,
                   **kwargs) -> None:
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    if requires is None:
        requires = []
    if not isinstance(model_groups, (list, tuple)):
        model_groups = [model_groups]
    if function_kwargs is None:
        function_kwargs = {}
    model_info = {
        'architectures': architectures,
        'model_groups': model_groups,
        'template': template,
        'requires': requires,
        'ignore_file_pattern': ignore_file_pattern,
        'support_flash_attn': support_flash_attn,
        'support_vllm': support_vllm,
        'support_lmdeploy': support_lmdeploy,
        'is_multimodal': is_multimodal,
        'is_moe': is_moe,
        **kwargs
    }

    if len(function_kwargs) > 0:
        get_function = partial(get_function, **function_kwargs)
    model_info['get_function'] = get_function
    MODEL_MAPPING[model_type] = model_info
    return


def get_model_tokenizer_from_repo(model_dir: str,
                                  torch_dtype: Optional[torch.dtype],
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  tokenizer=None,
                                  automodel_class=AutoModelForCausalLM,
                                  **kwargs):
    """load from an independent repository"""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    HfConfigFactory.compat_zero3(model_config)
    quant_info = HfConfigFactory.get_quant_info(model_config)

    quant_method = quant_info.get('quant_info')
    if quantization_config is not None:
        quant_method = quantization_config.quant_method
        # quant_bits = quantization_config.bits

    is_training = kwargs.pop('is_training', False)

    # if quant_method == 'awq' and is_training:
    #     check_awq_ext()
    # if quant_method == 'gptq' and is_training:
    #     patch_gptq_model(quant_bits, model_config, model_kwargs)

    if torch_dtype is not None:
        model_config.torch_dtype = torch_dtype

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    patch_rope_scaling(model_config, kwargs.pop('rope_scaling', None), kwargs.pop('max_length', None))
    if load_model:
        if kwargs.get('use_unsloth', False):
            model, tokenizer = load_by_unsloth(model_dir, torch_dtype, **kwargs)
        else:
            logger.info(f'model_kwargs: {model_kwargs}')
            model = load_by_transformers(automodel_class, model_dir, model_config, torch_dtype, quant_method == 'aqlm',
                                         is_training, model_kwargs, **kwargs)
    else:
        model = None
    tokenizer.config = model_config
    return model, tokenizer


def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        torch_dtype: torch.dtype,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        model_config=None,
                                        **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config_list = [model_config]
    for k in ['language_config', 'llm_config', 'text_config']:
        llm_config = getattr(model_config, k, None)
        if llm_config:
            config_list.append(llm_config)
            break
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    for config in config_list:
        attn_type.update_config(config)
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


def fix_transformers_upgrade(module: PreTrainedModel) -> None:
    # from 4.35, transformers changes its arguments of _set_gradient_checkpointing
    if version.parse(transformers.__version__) >= version.parse('4.35'):
        if isinstance(module, PreTrainedModel) and hasattr(module, '_set_gradient_checkpointing') \
                and 'value' in inspect.signature(module._set_gradient_checkpointing).parameters.keys():
            module._set_gradient_checkpointing = MethodType(PreTrainedModel._set_gradient_checkpointing, module)


def fix_gradient_checkpointing_warning(is_moe: bool = False) -> None:
    torch_version = version.parse(torch.__version__)
    if torch_version < version.parse('2'):
        return
    elif torch_version < version.parse('2.1'):
        # fix https://github.com/Dao-AILab/flash-attention/issues/341
        _use_reentrant = True
    else:
        _use_reentrant = is_moe
    if hasattr(torch.utils.checkpoint, '_checkpoint_origin'):
        return
    # fix torch
    _checkpoint_origin = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint._checkpoint_origin = _checkpoint_origin
    checkpoint = update_wrapper(
        lambda *args, use_reentrant=_use_reentrant, **kwargs: _checkpoint_origin(
            *args, use_reentrant=use_reentrant, **kwargs),
        _checkpoint_origin)
    torch.utils.checkpoint.checkpoint = checkpoint

    try:
        # fix gradient_checkpointing_enable
        import transformers.modeling_utils
        if hasattr(transformers.modeling_utils, 'checkpoint'):
            transformers.modeling_utils.checkpoint = checkpoint
    except ImportError:
        pass


def fix_do_sample_warning(generation_config) -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.do_sample is False:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50


def get_model_tokenizer(model_id_or_path: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        device_map: Union[str, Dict[str, Any], None] = 'auto',
                        model_kwargs: Optional[Dict[str, Any]] = None,
                        load_model: bool = True,
                        *,
                        use_hf: Optional[bool] = None,
                        model_type: Optional[str] = None,
                        revision: Optional[str] = None,
                        is_training: bool = False,
                        download_model: Optional[bool] = None,
                        **kwargs) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """
    torch_dtype: If you use None, it will retrieve the torch_dtype from the config.json file.
        However, if torch.float32 is retrieved, torch.float16 will be used.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model
    model_dir = safe_snapshot_download(
        model_id_or_path, revision=revision, download_model=download_model, use_hf=use_hf)

    if load_model:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        kwargs['model_config'] = model_config
        if model_type is None:
            model_types = HfConfigFactory.get_matched_model_types(model_config, model_dir)
            if len(model_types) > 1:
                raise ValueError('Unable to obtain the accurate model_type based on the model architecture. '
                                 f'Please explicitly provide the model_type. Available model_types: {model_types}')
            model_type = model_types[0]
            logger.info(f'Setting model_type: {model_type}')
        if torch_dtype is None:
            # params_dtype
            torch_dtype = HfConfigFactory.get_torch_dtype(model_config)
            if torch_dtype in {torch.float32, None}:
                torch_dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16
            logger.info(f'Setting torch_dtype: {torch_dtype}')

        quant_info = HfConfigFactory.get_quant_info(model_config)
        if quant_info is not None:
            kwargs.update(quant_info)
        if use_torchacc():
            device_map = None
        model_kwargs['device_map'] = device_map
        kwargs['is_training'] = is_training

    kwargs['model_type'] = model_type
    model_info = MODEL_MAPPING[model_type]
    requires = model_info['requires']
    for require in requires:
        require_version(require)
    get_function = model_info.get('get_function', get_model_tokenizer_from_repo)
    model, tokenizer = get_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)

    is_multimodal = model_info['is_multimodal']
    is_moe = model_info['is_moe']
    for obj in [model, tokenizer]:
        if obj is None:
            continue
        obj.model_type = model_type
        obj.model_dir = model_dir
        obj.is_multimodal = is_multimodal
        obj.is_moe = is_moe

    if model is not None:
        fix_gradient_checkpointing_warning(is_moe)
        model.max_model_len = HfConfigFactory.get_max_model_len(model.config)
        logger.info(f'model.max_model_len: {model.max_model_len}')
        fix_transformers_upgrade(model)

        # generation_config
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        if not hasattr(model, 'generation_config') and os.path.isfile(generation_config_path):
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        # fix llama2 warning
        fix_do_sample_warning(model.generation_config)

    return model, tokenizer


def _get_model_names(model_groups: List[ModelGroup]) -> List[str]:
    res = set()
    for model_group in model_groups:
        for model in model_group.models:
            for key in ['ms_model_id', 'hf_model_id', 'model_path']:
                value = getattr(model, key)

                if isinstance(value, str):
                    model_name = value.rsplit('/', 1)[-1]
                    res.add(model_name)
    return list(res)


def get_arch_mapping() -> Dict[str, Dict[str, List[str]]]:
    global ARCH_MAPPING
    if ARCH_MAPPING is None:
        # arch(str) -> Dict[model_type(str), List[model_name(str)]]
        ARCH_MAPPING = {}
        for model_type, model_info in MODEL_MAPPING.items():
            arch = model_info['architectures']
            model_names = _get_model_names(model_info['model_groups'])
            if arch not in ARCH_MAPPING:
                ARCH_MAPPING[arch] = {}
            ARCH_MAPPING[arch][model_type] = model_names
    return ARCH_MAPPING


def get_default_template_type(model_type: str) -> Optional[str]:
    return MODEL_MAPPING[model_type].get('template')
