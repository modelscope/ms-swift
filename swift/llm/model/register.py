# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import platform
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from peft import PeftModel
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import (is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_mps_available,
                                is_torch_npu_available, strtobool)
from transformers.utils.versions import require_version

from swift.utils import get_dist_setting, get_logger, is_mp, is_unsloth_available, patch_getattr, use_torchacc
from .constant import ModelType
from .patcher import (patch_automodel, patch_automodel_for_sequence_classification, patch_get_dynamic_module,
                      patch_mp_ddp, patch_tp_plan)
from .utils import AttnImpl, HfConfigFactory, InitModelStrategy, ModelInfo, safe_snapshot_download

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
        assert isinstance(self.models, (tuple, list)), f'self.models: {self.models}'


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
    ignore_patterns: Optional[List[str]] = None
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
        max_seq_length=args.max_length,
        full_finetuning=args.quant_bits is None,
        load_in_4bit=args.quant_bits == 4,
        load_in_8bit=args.quant_bits == 8,
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


def _patch_awq_compat(model_info):
    if version.parse(transformers.__version__) < version.parse('4.50') or model_info.quant_method != 'awq':
        return

    try:
        # compat transformers>=4.50 (autoawq)
        from transformers.quantizers.quantizer_awq import AwqQuantizer
        from transformers.integrations import get_keys_to_not_convert
        _process_model_before_weight_loading = AwqQuantizer._process_model_before_weight_loading

        def _new_process_model_before_weight_loading(self, model, *args, **kwargs):
            modules_to_not_convert = self.quantization_config.modules_to_not_convert
            if modules_to_not_convert is not None:
                self.quantization_config.modules_to_not_convert = list(
                    modules_to_not_convert) + get_keys_to_not_convert(model)
            return _process_model_before_weight_loading(self, model, *args, **kwargs)

        AwqQuantizer._process_model_before_weight_loading = _new_process_model_before_weight_loading
    except Exception:
        pass


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

    torch_dtype = model_info.torch_dtype
    model_config.torch_dtype = torch_dtype
    HfConfigFactory.compat_zero3(model_config)
    rope_scaling = kwargs.get('rope_scaling')
    if rope_scaling:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    num_labels = model_info.num_labels or getattr(model_config, 'num_labels', None)
    if num_labels and model_info.task_type in ['seq_cls', 'reranker']:
        model_info.num_labels = num_labels
        model_config.num_labels = num_labels

    model = None
    if load_model:
        _patch_awq_compat(model_info)
        logger.info(f'model_kwargs: {model_kwargs}')
        # fix seq_cls
        if model_info.task_type == 'seq_cls' and automodel_class is None:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
            except ValueError:
                model = None
        elif model_info.task_type == 'reranker' and automodel_class is None:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
            except ValueError:
                model = None

        automodel_class = automodel_class or AutoModelForCausalLM
        model_meta = kwargs['model_meta']
        if model is None:
            if model_info.task_type == 'seq_cls' and not model_meta.is_reward:
                context = partial(patch_automodel_for_sequence_classification, model_meta=model_meta)
            elif model_info.task_type == 'seq_cls' and model_meta.is_reward and model_config.num_labels > 1:
                logger.warning('You are using a reward model for seq_cls task and num_labels > 1, '
                               'ignore_mismatched_sizes will be set to True')
                model_kwargs['ignore_mismatched_sizes'] = True
                context = partial(patch_automodel_for_sequence_classification, model_meta=model_meta)
            elif model_info.task_type == 'reranker':
                # For reranker task, patch CausalLM to SequenceClassification with num_labels=1
                logger.info('Converting CausalLM to SequenceClassification for reranker task with num_labels=1')
                context = partial(patch_automodel_for_sequence_classification, model_meta=model_meta)
            elif model_info.task_type == 'generative_reranker':
                # For generative reranker, keep CausalLM structure unchanged
                logger.info('Loading model as CausalLM for generative_reranker task')
                context = partial(patch_automodel, automodel_class=automodel_class, model_info=model_info)
            else:
                context = partial(patch_automodel, automodel_class=automodel_class, model_info=model_info)
            with context():
                model = automodel_class.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)

        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = automodel_class.__name__

        if model_info.task_type == 'embedding' and automodel_class.__name__ != 'AutoModel':
            from swift.llm.model.patcher import patch_output_normalizer
            patch_output_normalizer(model, model_meta=model_meta)

        init_strategy = kwargs.get('init_strategy')
        if init_strategy is not None:
            InitModelStrategy.init_parameters(model, init_strategy)

    model_info.config = model_config if model is None else model.config

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
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', pad_token)

    return model, tokenizer


def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        model_info: ModelInfo,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        **kwargs):
    model_config = kwargs.get('model_config')
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'), kwargs.get('attn_impl_keys'))
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
        return 'auto' if is_mp() else f'npu:{local_rank}'
    elif is_torch_mps_available():
        return f'mps:{local_rank}'
    elif is_torch_cuda_available():
        return 'auto' if is_mp() else f'cuda:{local_rank}'
    else:
        return 'cpu'


def get_default_torch_dtype(torch_dtype: Optional[torch.dtype]):
    # torch_dtype: torch_dtype in config.json
    if torch_dtype is not None:
        return torch_dtype

    try:
        is_bf16_available = is_torch_bf16_gpu_available() or (is_torch_npu_available()
                                                              and torch.npu.is_bf16_supported())
    except:  # noqa
        is_bf16_available = False

    if is_torch_cuda_available() or is_torch_npu_available():
        if is_bf16_available:
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
        model_group = ModelMeta.get_matched_model_group(model_meta, model_name)
        if model_group is not None:
            model_meta = deepcopy(model_meta)
            for k, v in asdict(model_group).items():
                if v is not None and k in model_meta.__dict__:
                    setattr(model_meta, k, v)
            return model_meta


def _get_arch_mapping():
    res = {}
    for model_type, model_meta in MODEL_MAPPING.items():
        architectures = model_meta.architectures
        if not architectures:
            architectures.append('null')
        for arch in architectures:
            if arch not in res:
                res[arch] = []
            res[arch].append(model_type)
    return res


def get_matched_model_types(architectures: Optional[List[str]]) -> List[str]:
    """Get possible model_type."""
    architectures = architectures or ['null']
    if architectures:
        architectures = architectures[0]
    arch_mapping = _get_arch_mapping()
    return arch_mapping.get(architectures) or []


def _read_args_json_model_type(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'args.json')):
        return
    from swift.llm import BaseArguments
    args = BaseArguments.from_pretrained(model_dir)
    return args.model_type


def _get_model_info(model_dir: str, model_type: Optional[str], quantization_config) -> ModelInfo:
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        config = PretrainedConfig.get_config_dict(model_dir)[0]
    if quantization_config is not None:
        HfConfigFactory.set_config_attr(config, 'quantization_config', quantization_config)
    quant_info = HfConfigFactory.get_quant_info(config) or {}
    torch_dtype = HfConfigFactory.get_torch_dtype(config, quant_info)
    max_model_len = HfConfigFactory.get_max_model_len(config)
    rope_scaling = HfConfigFactory.get_config_attr(config, 'rope_scaling')

    if model_type is None:
        model_type = _read_args_json_model_type(model_dir)
    if model_type is None:
        architectures = HfConfigFactory.get_config_attr(config, 'architectures')
        model_types = get_matched_model_types(architectures)
        if len(model_types) > 1:
            raise ValueError('Please explicitly pass the model_type. For reference, '
                             f'the available model_types: {model_types}.')
        elif len(model_types) == 1:
            model_type = model_types[0]
    elif model_type not in MODEL_MAPPING:
        raise ValueError(f"model_type: '{model_type}' not in {list(MODEL_MAPPING.keys())}")

    res = ModelInfo(
        model_type,
        model_dir,
        torch_dtype,
        max_model_len,
        quant_info.get('quant_method'),
        quant_info.get('quant_bits'),
        rope_scaling=rope_scaling)
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
    if model_type is not None:
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
        if model_meta.task_type is not None:
            task_type = model_meta.task_type

    # Handle reranker task type
    if task_type == 'reranker':
        if num_labels is None:
            num_labels = 1  # Default to 1 for reranker tasks
        logger.info(f'Setting reranker task with num_labels={num_labels}')
    elif task_type == 'generative_reranker':
        # Generative reranker doesn't need num_labels as it uses CausalLM structure
        num_labels = None
        logger.info('Setting generative_reranker task (no num_labels needed)')
    elif task_type == 'seq_cls':
        assert num_labels is not None, 'Please pass the parameter `num_labels`.'

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
        max_memory: Union[str, Dict[str, Any]] = None,
        attn_impl: Literal['flash_attn', 'sdpa', 'eager', None] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        automodel_class=None,
        task_type: Literal['causal_lm', 'seq_cls', 'reranker', 'generative_reranker'] = None,
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
    if load_model:
        patch_mp_ddp()
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
    if max_memory:
        model_kwargs['max_memory'] = max_memory
    model_dir = model_info.model_dir
    get_function = model_meta.get_function
    kwargs['automodel_class'] = automodel_class
    kwargs['attn_impl'] = attn_impl
    kwargs['rope_scaling'] = rope_scaling
    kwargs['model_meta'] = model_meta
    with patch_get_dynamic_module(), patch_tp_plan(load_model):
        model, processor = get_function(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        patch_getattr(processor.__class__, 'tokenizer')
    else:
        tokenizer = processor
    problem_type = kwargs.get('problem_type')
    if problem_type is None and model_info.num_labels == 1:
        problem_type = 'regression'
    if problem_type is not None:
        model_info.config.problem_type = problem_type
    tokenizer.model_info = model_info
    tokenizer.model_meta = model_meta

    if model is not None:
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
