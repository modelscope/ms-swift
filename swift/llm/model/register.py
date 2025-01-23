# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import platform
import re
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from peft import PeftModel
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import (is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_mps_available,
                                is_torch_npu_available, strtobool)
from transformers.utils.versions import require_version

from swift.utils import get_dist_setting, get_logger, is_mp, is_unsloth_available, patch_getattr, use_torchacc
from .constant import ModelType
from .patcher import patch_automodel_for_awq, patch_automodel_for_sequence_classification
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

    # Higher priority. If set to None, the attributes of the ModelMeta will be used.
    ignore_patterns: Optional[List[str]] = None
    requires: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.models, (tuple, list)):
            self.models = [self.models]


@dataclass
class ModelMeta:
    model_type: Optional[str]
    # Used to list the model_ids from modelscope/huggingface,
    # which participate in the automatic inference of the model_type.
    model_groups: List[ModelGroup]
    template: Optional[str]
    get_function: GetModelTokenizerFunction

    model_arch: Optional[str] = None
    architectures: List[str] = field(default_factory=list)
    # Additional files that need to be saved for full parameter training/merge-lora.
    additional_saved_files: List[str] = field(default_factory=list)
    torch_dtype: Optional[torch.dtype] = None

    is_multimodal: bool = False
    is_reward: bool = False
    task_type: Optional[str] = None

    # File patterns to ignore when downloading the model.
    ignore_patterns: List[str] = field(default_factory=list)
    # Usually specifies the version limits of transformers.
    requires: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.template is None:
            self.template = 'dummy'
        if not isinstance(self.model_groups, (list, tuple)):
            self.model_groups = [self.model_groups]

    def get_matched_model_group(self, model_name: str) -> Optional[ModelGroup]:
        for model_group in self.model_groups:
            for model in model_group.models:
                for key in ['ms_model_id', 'hf_model_id', 'model_path']:
                    value = getattr(model, key)

                    if isinstance(value, str) and model_name == value.rsplit('/', 1)[-1].lower():
                        return model_group

    def check_requires(self, model_info=None):
        extra_requires = []
        if model_info and model_info.quant_method:
            mapping = {'bnb': ['bitsandbytes'], 'awq': ['autoawq'], 'gptq': ['auto_gptq'], 'aqlm': ['aqlm']}
            extra_requires += mapping.get(model_info.quant_method, [])
        requires = []
        for require in self.requires + extra_requires:
            try:
                require_version(require)
            except ImportError:
                requires.append(f'"{require}"')
        if requires:
            requires = ' '.join(requires)
            logger.warning(f'Please install the package: `pip install {requires} -U`.')


MODEL_MAPPING: Dict[str, ModelMeta] = {}


def register_model(model_meta: ModelMeta, *, exist_ok: bool = False) -> None:
    """
    model_type: The unique ID for the model type. Models with the same model_type share
        the same architectures, template, get_function, etc.
    """
    model_type = model_meta.model_type
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    from .constant import MLLMModelType, RMModelType
    if model_type in MLLMModelType.__dict__:
        model_meta.is_multimodal = True
    if model_type in RMModelType.__dict__:
        model_meta.is_reward = True
    MODEL_MAPPING[model_type] = model_meta


def load_by_unsloth(args):
    """Load model by unsloth"""
    assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`: `pip install unsloth`'
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'
    model_info = args.model_info
    model_meta = args.model_meta
    if model_meta.is_multimodal:
        from unsloth import FastVisionModel as UnslothModel
    else:
        from unsloth import FastLanguageModel as UnslothModel
    model, processor = UnslothModel.from_pretrained(
        model_name=args.adapters and args.adapters[0] or args.model_dir,
        dtype=args.torch_dtype,
        max_seq_length=model_info.max_model_len,
        load_in_4bit=args.quant_bits == 4,
        trust_remote_code=True,
    )
    if isinstance(model, PeftModel):
        base_model = model.model
    else:
        base_model = model
    base_model.model_dir = args.model_dir
    base_model.model_info = model_info
    base_model.model_meta = model_meta
    processor.model_info = model_info
    processor.model_meta = model_meta
    return model, processor


def get_model_tokenizer_from_local(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   *,
                                   tokenizer=None,
                                   model_config=None,
                                   automodel_class=None,
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
    rope_scaling = kwargs.get('rope_scaling')
    if rope_scaling is not None:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    num_labels = model_info.num_labels or getattr(model_config, 'num_labels', None)
    if num_labels and model_info.task_type != 'causal_lm':
        model_info.num_labels = num_labels
        model_config.num_labels = num_labels

    model = None
    if load_model:
        logger.info(f'model_kwargs: {model_kwargs}')
        # fix seq_cls
        if model_info.task_type == 'seq_cls' and automodel_class is None:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
            except ValueError:
                model = None

        automodel_class = automodel_class or AutoModelForCausalLM
        if model is None:
            if model_info.task_type == 'seq_cls':
                context = partial(patch_automodel_for_sequence_classification, model_meta=kwargs['model_meta'])
            elif 'AutoAWQFor' in automodel_class.__name__:
                context = patch_automodel_for_awq
            else:
                context = nullcontext
            with context():
                model = automodel_class.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)

        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = automodel_class.__name__
    return model, tokenizer


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
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    return model, processor


def get_model_tokenizer_reward_model(model_dir, *args, **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if 'AutoModel' in (getattr(model_config, 'auto_map', None) or {}):
        kwargs['automodel_class'] = AutoModel
    return get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)


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
    if local_rank == -1:
        local_rank = 0
    if is_torch_npu_available():
        return f'npu:{local_rank}'
    elif is_torch_mps_available():
        return f'mps:{local_rank}'
    elif is_torch_cuda_available():
        if is_mp():
            return 'auto'
        else:
            return f'cuda:{local_rank}'
    else:
        return 'cpu'


def get_default_torch_dtype(torch_dtype: Optional[torch.dtype]):
    # torch_dtype: torch_dtype in config.json
    if torch_dtype is not None:
        return torch_dtype

    if is_torch_cuda_available() or is_torch_npu_available():
        if is_torch_bf16_gpu_available():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        # cpu
        return torch.float32


def get_model_name(model_id_or_path: str) -> Optional[str]:
    assert isinstance(model_id_or_path, str), f'model_id_or_path: {model_id_or_path}'
    # compat hf hub
    model_id_or_path = model_id_or_path.rstrip('/')
    match_ = re.search('/models--.+?--(.+?)/snapshots/', model_id_or_path)
    if match_ is not None:
        return match_.group(1)

    model_name = model_id_or_path.rsplit('/', 1)[-1]
    if platform.system().lower() == 'windows':
        model_name = model_name.rsplit('\\', 1)[-1]
    # compat modelscope snapshot_download
    model_name = model_name.replace('___', '.')
    return model_name


def get_all_models() -> List[str]:
    use_hf = strtobool(os.environ.get('USE_HF', 'False'))
    models = []
    for model_type in ModelType.get_model_name_list():
        model_meta = MODEL_MAPPING.get(model_type)
        if model_meta:
            for group in model_meta.model_groups:
                for model in group.models:
                    if use_hf:
                        if model.hf_model_id:
                            models.append(model.hf_model_id)
                    else:
                        if model.ms_model_id:
                            models.append(model.ms_model_id)
    return models


def get_matched_model_meta(model_id_or_path: str) -> Optional[ModelMeta]:
    model_name = get_model_name(model_id_or_path).lower()
    for model_type, model_meta in MODEL_MAPPING.items():
        model_group = model_meta.get_matched_model_group(model_name)
        if model_group is not None:
            model_meta = deepcopy(model_meta)
            for k, v in asdict(model_group).items():
                if v is not None and k in model_meta.__dict__:
                    setattr(model_meta, k, v)
            return model_meta


def _get_model_info(model_dir: str, model_type: Optional[str], quantization_config) -> ModelInfo:
    config_dict = PretrainedConfig.get_config_dict(model_dir)[0]
    if quantization_config is not None:
        config_dict['quantization_config'] = quantization_config
    quant_info = HfConfigFactory.get_quant_info(config_dict) or {}
    torch_dtype = HfConfigFactory.get_torch_dtype(config_dict, quant_info)
    max_model_len = HfConfigFactory.get_max_model_len(config_dict)
    rope_scaling = HfConfigFactory.get_config_attr(config_dict, 'rope_scaling')

    if model_type is None:
        model_types = HfConfigFactory.get_matched_model_types(config_dict)  # config.json
        if len(model_types) > 1:
            raise ValueError('Please explicitly pass the model_type. For reference, '
                             f'the available model_types: {model_types}.')
        elif len(model_types) == 1:
            model_type = model_types[0]
    elif model_type not in MODEL_MAPPING:
        raise ValueError(f"model_type: '{model_type}' not in {list(MODEL_MAPPING.keys())}")

    res = ModelInfo(model_type, model_dir, torch_dtype, max_model_len, quant_info.get('quant_method'),
                    quant_info.get('quant_bits'), rope_scaling)
    return res


def get_model_info_meta(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        # hub
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        download_model: bool = False,
        # model kwargs
        model_type: Optional[str] = None,
        quantization_config=None,
        task_type=None,
        num_labels=None,
        **kwargs) -> Tuple[ModelInfo, ModelMeta]:
    model_meta = get_matched_model_meta(model_id_or_path)
    model_dir = safe_snapshot_download(
        model_id_or_path,
        revision=revision,
        download_model=download_model,
        use_hf=use_hf,
        ignore_patterns=getattr(model_meta, 'ignore_patterns', None),
        hub_token=hub_token)

    model_type = model_type or getattr(model_meta, 'model_type', None)
    model_info = _get_model_info(model_dir, model_type, quantization_config=quantization_config)
    if model_type is None and model_info.model_type is not None:
        model_type = model_info.model_type
        logger.info(f'Setting model_type: {model_type}')
    if model_meta is None and model_type is not None:
        model_meta = MODEL_MAPPING[model_type]
    if model_meta is None:
        model_meta = ModelMeta(None, [], 'dummy', get_model_tokenizer_from_local, model_arch=None)
        logger.info(f'Temporarily create model_meta: {model_meta}')

    if torch_dtype is None:
        torch_dtype = model_meta.torch_dtype or get_default_torch_dtype(model_info.torch_dtype)
        logger.info(f'Setting torch_dtype: {torch_dtype}')
    model_info.torch_dtype = torch_dtype
    if task_type is None:
        if model_meta.is_reward:
            num_labels = 1
        if num_labels is None:
            task_type = 'causal_lm'
        else:
            task_type = 'seq_cls'
        if task_type == 'seq_cls':
            assert num_labels is not None, 'Please pass the parameter `num_labels`.'
        if model_meta.task_type is not None:
            task_type = model_meta.task_type
    model_info.task_type = task_type
    model_info.num_labels = num_labels

    model_meta.check_requires(model_info)
    return model_info, model_meta


def get_model_tokenizer(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, Dict[str, Any], None] = None,
        *,
        load_model: bool = True,
        # hub
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        download_model: Optional[bool] = None,
        # model kwargs
        model_type: Optional[str] = None,
        quantization_config=None,
        attn_impl: Literal['flash_attn', 'sdpa', 'eager', None] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        automodel_class=None,
        task_type: Literal['causal_lm', 'seq_cls'] = None,
        num_labels: Optional[int] = None,
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

    model_info, model_meta = get_model_info_meta(
        model_id_or_path,
        torch_dtype,
        use_hf=use_hf,
        hub_token=hub_token,
        revision=revision,
        download_model=download_model,
        model_type=model_type,
        quantization_config=quantization_config,
        task_type=task_type,
        num_labels=num_labels)

    if not use_torchacc() and device_map is None:
        device_map = get_default_device_map()
    model_kwargs['device_map'] = device_map
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    model_dir = model_info.model_dir
    get_function = model_meta.get_function
    kwargs['automodel_class'] = automodel_class
    kwargs['attn_impl'] = attn_impl
    kwargs['rope_scaling'] = rope_scaling
    kwargs['model_meta'] = model_meta
    model, processor = get_function(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        patch_getattr(processor.__class__, 'tokenizer')
    else:
        tokenizer = processor
    tokenizer.model_info = model_info
    tokenizer.model_meta = model_meta

    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None

    if model is not None:
        # fix seq classification task
        pad_token_id = model.config.pad_token_id or tokenizer.pad_token_id
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', pad_token_id)
        model.model_info = model_info
        model.model_meta = model_meta
        model.model_dir = model_dir

        # generation_config
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        if not hasattr(model, 'generation_config') and os.path.isfile(generation_config_path):
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        # fix llama2 warning
        if getattr(model, 'generation_config', None):
            fix_do_sample_warning(model.generation_config)
    return model, processor
