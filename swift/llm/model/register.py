# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import itertools
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import transformers
from packaging import version
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizerBase)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_npu_available
from transformers.utils.versions import require_version

from swift.utils import get_dist_setting, get_logger, is_ddp_plus_mp, is_dist, is_unsloth_available, use_torchacc
from .utils import AttnImpl, HfConfigFactory, ModelInfo, safe_snapshot_download

GetModelTokenizerFunction = Callable[..., Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]]
logger = get_logger()


@dataclass
class Model:
    ms_model_id: Optional[str] = None
    hf_model_id: Optional[str] = None
    model_path: Optional[str] = None

    ms_revision: Optional[str] = None
    hf_revision: Optional[str] = None


@dataclass
class ModelGroup:
    models: List[Model]
    tags: List[str] = field(default_factory=list)

    # Higher priority. If set to None, the attributes of the DatasetMeta will be used.
    ignore_file_pattern: Optional[List[str]] = None
    requires: Optional[List[str]] = None
    support_flash_attn: Optional[bool] = None
    support_vllm: Optional[bool] = None
    support_lmdeploy: Optional[bool] = None
    support_megatron: Optional[bool] = None


@dataclass
class ModelMeta:
    model_type: str
    # Used to list the model_ids from huggingface/modelscope,
    # which participate in the automatic inference of the model_type.
    model_groups: List[ModelGroup]
    template: str
    get_function: GetModelTokenizerFunction

    model_arch: Optional[str]
    architectures: List[str] = field(default_factory=list)
    is_moe: bool = False
    is_multimodal: bool = False
    # Additional files that need to be saved for full parameter training/merge-lora.
    additional_saved_files: List[str] = field(default_factory=list)
    support_gradient_checkpointing: bool = True
    torch_dtype: Optional[torch.dtype] = None

    # File patterns to ignore when downloading the model.
    ignore_file_pattern: List[str] = field(default_factory=list)
    # Usually specifies the version limits of transformers.
    requires: List[str] = field(default_factory=list)

    def get_matched_model_group(self, model_name: str) -> Optional[ModelGroup]:
        for model_group in self.model_groups:
            for model in model_group.models:
                for key in ['ms_model_id', 'hf_model_id', 'model_path']:
                    value = getattr(model, key)

                    if isinstance(value, str) and model_name == value.rsplit('/', 1)[-1].lower():
                        return model_group

    def check_requires(self):
        # TODO: error to warning
        for require in self.requires:
            require_version(require)

    def check_flash_attn(self, attn_impl: Optional[str]) -> None:
        from .utils import AttnImpl
        if attn_impl is None:
            return
        if attn_impl == AttnImpl.flash_attn and not self.support_flash_attn:
            logger.warning(f'attn_impl: {attn_impl}, but support_flash_attn: {self.support_flash_attn}')

    def check_infer_backend(self, infer_backend: str) -> None:
        if infer_backend == 'vllm' and not self.support_vllm:
            logger.warning(f'infer_backend: {infer_backend}, but support_vllm: {self.support_vllm}')
        elif infer_backend == 'lmdeploy' and not self.support_lmdeploy:
            logger.warning(f'infer_backend: {infer_backend}, but support_lmdeploy: {self.support_lmdeploy}')

    def check_gradient_checkpointing(self, gradient_checkpoint: bool) -> None:
        if gradient_checkpoint and not self.support_gradient_checkpointing:
            logger.warning(f'gradient_checkpoint: {gradient_checkpoint}, but support_gradient_checkpointing: '
                           f'{self.support_gradient_checkpointing}')


MODEL_MAPPING: Dict[str, ModelMeta] = {}


# [TODO:eos_token -> template]
def register_model(model_meta: ModelMeta, *, exist_ok: bool = False) -> None:
    """
    model_type: The unique ID for the model type. Models with the same model_type share
        the same architectures, template, get_function, etc.
    """
    model_type = model_meta.model_type
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    from .constant import MLLMModelType
    if model_type in MLLMModelType.__dict__:
        model_meta.is_multimodal = True
    MODEL_MAPPING[model_type] = model_meta


def load_by_unsloth(model_dir: str,
                    torch_dtype: torch.dtype,
                    max_seq_length: Optional[int] = None,
                    load_in_4bit: bool = True):
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
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   *,
                                   tokenizer=None,
                                   model_config=None,
                                   automodel_class=AutoModelForCausalLM,
                                   **kwargs):
    """Load the model and tokenizer from the local model_dir."""

    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # fix prediction_step (internvl2, ovis, ...)
    if not hasattr(model_config, 'keys_to_ignore_at_inference'):
        model_config.keys_to_ignore_at_inference = []
    if 'past_key_values' not in model_config.keys_to_ignore_at_inference:
        model_config.keys_to_ignore_at_inference.append('past_key_values')
    model_info.config = model_config

    torch_dtype = model_info.torch_dtype
    model_config.torch_dtype = torch_dtype
    HfConfigFactory.compat_zero3(model_config)
    if model_info.rope_scaling is not None:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', model_info.rope_scaling)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = None
    if load_model:
        if kwargs.get('use_unsloth', False):
            unsloth_kwargs = kwargs.get('unsloth_kwargs') or {}
            logger.info(f'unsloth_kwargs: {unsloth_kwargs}')
            model, tokenizer = load_by_unsloth(model_dir, torch_dtype, **unsloth_kwargs)
        else:
            logger.info(f'model_kwargs: {model_kwargs}')
            model = automodel_class.from_pretrained(
                model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)

    # fix not save modeling_xxx.py (transformers 4.45)
    # https://github.com/huggingface/transformers/issues/24737
    has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
    if model is not None and has_remote_code and model._auto_class is None:
        model._auto_class = automodel_class.__name__
    return model, tokenizer


def get_model_with_value_head(model) -> 'AutoModelForCausalLMWithValueHead':
    from trl import AutoModelForCausalLMWithValueHead
    lm_head_namings = ['lm_head', 'embed_out']
    if not any(hasattr(model, attribute) for attribute in lm_head_namings):
        setattr(model, 'lm_head', None)  # avoid ValueError

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    def patch_valuehead_model(model):
        attr_list = [
            'get_input_embeddings', 'vis_processor', 'extract_feature', 'get_rope_index', 'model', 'vision_tower',
            'img2emb', '_encode_image', '_merge_input_ids_with_image_features', 'prepare_inputs_embeds',
            'build_conversation_input_ids', 'config', 'get_slice_image_placeholder', 'transform', 'get_vllm_embedding',
            'forward_image', 'dtype', 'base_model_prefix', 'device', 'visual'
        ]
        for attr in attr_list:
            if hasattr(model.pretrained_model, attr) and not hasattr(model, attr):
                setattr(model, attr, getattr(model.pretrained_model, attr))

        # PPO compatible
        if not hasattr(model, 'score'):
            setattr(model, 'score', model.v_head)
        if model.base_model_prefix == '' and hasattr(model.pretrained_model, 'language_model'):
            model.base_model_prefix = model.pretrained_model.language_model.base_model_prefix

        base_model_prefix = model.pretrained_model.base_model_prefix
        if hasattr(model.pretrained_model, base_model_prefix):
            setattr(model, base_model_prefix, getattr(model.pretrained_model, base_model_prefix))

    patch_valuehead_model(model)

    # try to load local vhead weights
    vhead_params = None
    try:
        from safetensors import safe_open
        vhead_file = os.path.join(model.pretrained_model.model_dir, 'value_head.safetensors')
        with safe_open(vhead_file, framework='pt', device='cpu') as f:
            vhead_params = {key: f.get_tensor(key) for key in f.keys()}
    except Exception:
        pass

    try:
        vhead_file = os.path.join(model.pretrained_model.model_dir, 'value_head.bin')
        vhead_params = torch.load(vhead_file, map_location='cpu')
    except Exception:
        pass

    if vhead_params is not None:
        model.load_state_dict(vhead_params, strict=False)
        logger.info(f'Loading value head weights from {vhead_file}')
    else:
        logger.info('The local value head weight file was not detected.'
                    'Ignore it if this is during the reward modeling phase,')
    return model


def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        model_info: ModelInfo,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        **kwargs):
    model_config = kwargs.get('model_config')
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'))
    kwargs['model_config'] = model_config
    return get_model_tokenizer_from_local(model_dir, model_info, model_kwargs, load_model, **kwargs)


def get_model_tokenizer_multimodal(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['tokenizer'] = processor.tokenizer
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    return model, processor


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


def fix_do_sample_warning(generation_config: GenerationConfig) -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.temperature == 0:
        generation_config.do_sample = False
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


def _check_torch_dtype(torch_dtype: torch.dtype):
    if is_torch_cuda_available() or is_torch_npu_available():

        if torch_dtype == torch.bfloat16:
            support_bf16 = is_torch_bf16_gpu_available()
            if not support_bf16:
                logger.warning(f'torch_dtype: {torch_dtype}, but support_bf16: {support_bf16}.')
    else:
        # cpu
        if torch_dtype == torch.float16:
            logger.warning(f'torch_dtype: {torch_dtype}. The CPU does not support matrix multiplication with FP16.')


def get_default_torch_dtype(torch_dtype: Optional[torch.dtype]):
    # torch_dtype: torch_dtype in config.json
    if is_torch_cuda_available() or is_torch_npu_available():
        if is_torch_bf16_gpu_available():
            if torch_dtype in {torch.float16, torch.bfloat16}:
                res = torch_dtype
            else:
                res = torch.bfloat16
        else:
            res = torch.float16
    else:
        # cpu
        res = torch.float32
    return res


def _get_model_name(model_id_or_path: str) -> str:
    # compat hf hub
    match_ = re.search('/models--.+?--(.+?)/snapshots/', model_id_or_path)
    if match_ is not None:
        model_name = match_.group(1)
    else:
        model_name = model_id_or_path.rsplit('/', 1)[-1]
    return model_name.lower()


def get_matched_model_meta(model_id_or_path: str) -> Optional[ModelMeta]:
    assert isinstance(model_id_or_path, str), f'model_id_or_path: {model_id_or_path}'
    model_name = _get_model_name(model_id_or_path).lower()
    for model_type, model_meta in MODEL_MAPPING.items():
        model_group = model_meta.get_matched_model_group(model_name)
        if model_group is not None:
            model_meta = deepcopy(model_meta)
            for k, v in asdict(model_group).items():
                if v is not None and k in model_meta.__dict__:
                    setattr(model_meta, k, v)
            return model_meta


def get_model_info(model_dir: str,
                   model_type: Optional[str],
                   quantization_config,
                   attn_impl: AttnImpl,
                   rope_scaling: Optional[Dict[str, Any]] = None) -> ModelInfo:
    config_dict = PretrainedConfig.get_config_dict(model_dir)[0]
    if quantization_config is None:
        config_dict['quantization_config'] = quantization_config
    quant_info = HfConfigFactory.get_quant_info(config_dict) or {}
    torch_dtype = HfConfigFactory.get_torch_dtype(config_dict, quant_info)
    max_model_len = HfConfigFactory.get_max_model_len(config_dict)

    if model_type is None:
        model_types = HfConfigFactory.get_matched_model_types(config_dict)  # config.json
        if len(model_types) > 1:
            raise ValueError('Please explicitly pass the model_type. For reference, '
                             f'the available model_types: {model_types}.')
        elif len(model_types) == 1:
            model_type = model_types[0]

    res = ModelInfo(model_type, model_dir, torch_dtype, max_model_len, quant_info.get('quant_method'),
                    quant_info.get('quant_bits'), attn_impl, rope_scaling)
    return res


def patch_processor(processor):
    if hasattr(processor, '_patch'):
        return

    def __getattr__(self, key: str):
        try:
            return super(processor.__class__, self).__getattr__(key)
        except AttributeError:
            if 'tokenizer' in self.__dict__:
                return getattr(self.tokenizer, key)
            raise

    processor.__class__.__getattr__ = __getattr__
    processor.__class__._patch = True


def get_model_tokenizer(model_id_or_path: str,
                        torch_dtype: Optional[torch.dtype] = None,
                        device_map: Union[str, Dict[str, Any], None] = None,
                        load_model: bool = True,
                        *,
                        quantization_config=None,
                        model_type: Optional[str] = None,
                        attn_impl: Literal['flash_attn', 'sdpa', 'eager', None] = None,
                        rope_scaling: Optional[Dict[str, Any]] = None,
                        use_hf: Optional[bool] = None,
                        revision: Optional[str] = None,
                        download_model: Optional[bool] = None,
                        automodel_class=AutoModelForCausalLM,
                        model_kwargs: Optional[Dict[str, Any]] = None,
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
        If set to None : It will be automatically selected between sdpa and eager.
    download_model: Whether to download the model weights. If `None`, it will be selected based on load_model.
    """

    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model
    ignore_file_pattern = ['*.zip', '*.gguf', '*.pth', '*.pt', 'consolidated*']
    model_meta = get_matched_model_meta(model_id_or_path)
    if getattr(model_meta, 'ignore_file_pattern', None) is not None:
        ignore_file_pattern += model_meta.ignore_file_pattern

    model_dir = safe_snapshot_download(
        model_id_or_path,
        revision=revision,
        download_model=download_model,
        use_hf=use_hf,
        ignore_file_pattern=ignore_file_pattern)

    if not use_torchacc() and device_map is None:
        device_map = get_default_device_map()
    model_kwargs['device_map'] = device_map
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config

    model_info = get_model_info(
        model_dir,
        getattr(model_meta, 'model_type', None),
        quantization_config=quantization_config,
        attn_impl=attn_impl,
        rope_scaling=rope_scaling)
    if model_type is None and model_info.model_type is not None:
        model_type = model_info.model_type
        logger.info(f'Setting model_type: {model_type}')
    if model_meta is None and model_type is not None:
        model_meta = MODEL_MAPPING[model_type]
    if model_meta is None:
        model_meta = ModelMeta('', [], '', get_model_tokenizer_from_local, model_arch=None)
        logger.info(f'Temporarily create model_meta: {model_meta}')

    if torch_dtype is None:
        torch_dtype = model_meta.torch_dtype or get_default_torch_dtype(model_info.torch_dtype)
        logger.info(f'Setting torch_dtype: {torch_dtype}')
    _check_torch_dtype(torch_dtype)
    model_info.torch_dtype = torch_dtype

    model_meta.check_requires()
    model_meta.check_flash_attn(attn_impl)
    get_function = model_meta.get_function
    kwargs['automodel_class'] = automodel_class
    model, tokenizer = get_function(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, 'tokenizer'):
        patch_processor(tokenizer)
    tokenizer.model_info = model_info
    tokenizer.model_meta = model_meta

    if model is not None:
        model.model_info = model_info
        model.model_meta = model_meta
        model.model_dir = model_dir
        fix_gradient_checkpointing_warning(model_meta.is_moe)
        fix_transformers_upgrade(model)

        # generation_config
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        # TODO:model.llm.generation_config: deepseek-vl
        if not hasattr(model, 'generation_config') and os.path.isfile(generation_config_path):
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        # fix llama2 warning
        fix_do_sample_warning(model.generation_config)
    return model, tokenizer
