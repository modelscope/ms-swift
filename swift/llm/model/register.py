import inspect
import os
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available
from transformers.utils.versions import require_version

from swift.llm import TemplateType
from swift.utils import get_dist_setting, get_logger, is_ddp_plus_mp, is_dist, is_unsloth_available, use_torchacc
from .utils import AttnImpl, HfConfigFactory, safe_snapshot_download

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
                   additional_saved_files: Optional[List[str]] = None,
                   is_multimodal: bool = False,
                   is_moe: bool = False,
                   support_flash_attn: bool = False,
                   support_vllm: bool = False,
                   support_lmdeploy: bool = False,
                   function_kwargs: Optional[Dict[str, Any]] = None,
                   exist_ok: bool = False,
                   **kwargs) -> None:
    """
    model_type: The unique ID for the model type. Models with the same model_type share
        the same architectures, template, get_function, etc.
    architectures: Used to automatically infer the model_type from config.json.
    model_groups: Used to list the model_ids from huggingface/modelscope,
        which participate in the automatic inference of the model_type.
    template: chat_template & generation_template. This will be determined based on
        whether the `swift pt` command is used to start.
    get_function: A function to obtain the model and tokenizer based on model_dir.

    requires: Usually specifies the version limits of transformers.
    ignore_file_pattern: File patterns to ignore when downloading the model.
    additional_saved_files: Additional files that need to be saved for full parameter training/merge-lora.
    is_multimodal: Whether it is a multimodal model.
    is_moe: Whether it is a moe model.
    support_flash_attn: Whether it supports flash attention.
    support_vllm: Whether it supports vllm inference acceleration.
    support_lmdeploy: Whether it supports lmdeploy inference acceleration.
    """
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    if requires is None:
        requires = []
    if additional_saved_files is None:
        additional_saved_files = []
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
        'additional_saved_files': additional_saved_files,
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


def load_by_unsloth(model_dir, torch_dtype, max_seq_length: int = None, load_in_4bit: bool = True):
    """Load model by unsloth"""
    # TODO:check
    assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`'
    from unsloth import FastLanguageModel
    return FastLanguageModel.from_pretrained(
        model_name=model_dir,
        dtype=torch_dtype,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )


def get_model_tokenizer_from_local(model_dir: str,
                                   torch_dtype: Optional[torch.dtype],
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   model_config=None,
                                   tokenizer=None,
                                   automodel_class=AutoModelForCausalLM,
                                   **kwargs):
    """Load the model and tokenizer from the local model_dir."""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    HfConfigFactory.compat_zero3(model_config)

    if torch_dtype is not None:
        model_config.torch_dtype = torch_dtype
    rope_scaling = kwargs.get('rope_scaling', None)
    if rope_scaling is not None:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if load_model:
        if kwargs.get('use_unsloth', False):
            unsloth_kwargs = kwargs.get('unsloth_kwargs') or {}
            logger.info(f'unsloth_kwargs: {unsloth_kwargs}')
            model, tokenizer = load_by_unsloth(model_dir, torch_dtype, **unsloth_kwargs)
        else:
            logger.info(f'model_kwargs: {model_kwargs}')
            model = automodel_class.from_pretrained(
                model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
    else:
        model = None
    tokenizer.config = model_config
    model.quant_method = kwargs.get('quant_method')
    model.quant_bits = kwargs.get('bits')
    model.is_training = kwargs.get('is_training', False)
    max_model_len = HfConfigFactory.get_max_model_len(model_config)
    model.max_model_len = max_model_len
    logger.info(f'model.max_model_len: {max_model_len}')
    return model, tokenizer


def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        torch_dtype: torch.dtype,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        model_config=None,
                                        **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl', 'auto'))
    return get_model_tokenizer_from_local(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


def get_model_tokenizer_multimodal(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


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


def get_default_device_map():
    if is_deepspeed_zero3_enabled() or os.environ.get('ACCELERATE_USE_FSDP', 'False') == 'true':
        return None
    local_rank = get_dist_setting()[1]
    if is_torch_npu_available():
        if local_rank >= 0:
            return f'npu:{local_rank}'
        else:
            return 'npu:0'
    if torch.cuda.device_count() == 0:
        return 'cpu'
    elif torch.cuda.device_count() == 1:
        return 'cuda:0'
    elif is_dist() and not is_ddp_plus_mp():
        return f'cuda:{local_rank}'
    else:
        return 'auto'


def get_model_tokenizer(model_id_or_path: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        model_kwargs: Optional[Dict[str, Any]] = None,
                        load_model: bool = True,
                        *,
                        use_hf: Optional[bool] = None,
                        model_type: Optional[str] = None,
                        attn_impl: Literal['flash_attn', 'sdpa', 'eager', 'auto'] = 'auto',
                        download_model: Optional[bool] = None,
                        **kwargs) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """
    model_id_or_path: The path to the model or the model_id from modelscope/huggingface (controlled by `use_hf`).
    torch_dtype: If you pass `None`, it will retrieve the torch_dtype from the config.json file.
    model_kwargs: Passed to `automodel_class.from_pretrained`.
    load_model: Whether to load the model. If set to False, the model will return `None`.
    use_hf: Indicates whether the model download hub is modelscope or huggingface.
    model_type: If it is not possible to uniquely determine the model_type from the architecture in config.json,
        it needs to be provided.
    attn_impl: If set to 'flash_attn': It will automatically convert names based on the model.
        If set to 'auto': It will be automatically selected between sdpa and eager.
    download_model: Whether to download the model weights. If `None`, it will be selected based on load_model.
    """

    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model
    revision = kwargs.pop('revision', None)
    model_dir = safe_snapshot_download(
        model_id_or_path, revision=revision, download_model=download_model, use_hf=use_hf)

    if load_model:
        if use_torchacc():
            model_kwargs['device_map'] = None
        elif 'device_map' not in model_kwargs:
            model_kwargs['device_map'] = get_default_device_map()
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        kwargs['model_config'] = model_config
        if model_type is None:
            model_types = HfConfigFactory.get_matched_model_types(model_config, model_dir)
            if len(model_types) > 1:
                raise ValueError('Unable to obtain the accurate model_type based on the model architecture. '
                                 f'Please explicitly provide the model_type. Available model_types: {model_types}')
            model_type = model_types[0]
            logger.info(f'Setting model_type: {model_type}')
        quant_info = HfConfigFactory.get_quant_info(model_config)
        if torch_dtype is None:
            torch_dtype = HfConfigFactory.get_torch_dtype(model_config)
            if torch_dtype is None:
                torch_dtype = quant_info.get('torch_dtype')
            if torch_dtype in {torch.float32, None}:
                torch_dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16

            logger.info(f'Setting torch_dtype: {torch_dtype}')

        if quant_info is not None:
            quant_info.pop('torch_dtype', None)
            kwargs.update(quant_info)

    kwargs.update({'model_type': model_type, 'attn_impl': attn_impl})
    model_info = MODEL_MAPPING[model_type]
    requires = model_info['requires']
    for require in requires:
        require_version(require)
    get_function = model_info.get('get_function', get_model_tokenizer_from_local)
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


class ModelInfoReader:
    # info
    @staticmethod
    def get_default_template_type(model_type: str) -> Optional[str]:
        return MODEL_MAPPING[model_type].get('template')
