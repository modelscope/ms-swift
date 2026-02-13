# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from types import MethodType
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from peft import PeftModel
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import strtobool

from swift.utils import (HfConfigFactory, Processor, get_generative_reranker_logits, get_logger, is_unsloth_available,
                         patch_getattr)
from .constant import ModelType
from .model_meta import MODEL_MAPPING, BaseModelLoader, ModelInfo, ModelMeta, get_model_info_meta
from .patcher import (get_lm_head_model, patch_attach_align_device_hook_on_blocks, patch_automodel,
                      patch_automodel_for_sequence_classification, patch_get_dynamic_module, patch_module_forward,
                      patch_mp_ddp, patch_tp_plan)
from .utils import AttnImpl, InitModelStrategy, get_default_device_map

logger = get_logger()

transformers_5 = version.parse(transformers.__version__) >= version.parse('5.0.0.dev')


def register_model(model_meta: ModelMeta, *, exist_ok: bool = False) -> None:
    """
    model_type: The unique ID for the model type. Models with the same model_type share
        the same architectures, template, get_function, etc.
    """
    from .model_arch import get_model_arch
    model_type = model_meta.model_type
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    if model_meta.model_arch:
        model_meta.model_arch = get_model_arch(model_meta.model_arch)
    MODEL_MAPPING[model_type] = model_meta


def load_by_unsloth(args):
    """Load model by unsloth"""
    assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`: `pip install unsloth`'
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'
    model_info = args.model_info
    model_meta = args.model_meta

    os.environ['UNSLOTH_IS_PRESENT'] = '1'

    @contextmanager
    def _patch_distributed_function():
        from unsloth_zoo import utils, compiler

        def distributed_function(n=1, function=None, *args, **kwargs):
            return function(*args, **kwargs)

        _origin_distributed_function = utils.distributed_function
        utils.distributed_function = distributed_function
        compiler.distributed_function = distributed_function
        yield
        utils.distributed_function = _origin_distributed_function
        compiler.distributed_function = _origin_distributed_function

    with _patch_distributed_function():
        if model_meta.is_multimodal:
            from unsloth import FastVisionModel as UnslothModel
        elif model_info.is_moe_model:
            from unsloth import FastModel as UnslothModel
        else:
            from unsloth import FastLanguageModel as UnslothModel

        model, processor = UnslothModel.from_pretrained(
            model_name=args.adapters and args.adapters[0] or args.model_dir,
            dtype=args.torch_dtype,
            max_seq_length=args.max_length,
            full_finetuning=args.tuner_type == 'full',
            load_in_4bit=args.quant_bits == 4,
            load_in_8bit=args.quant_bits == 8,
            device_map=args.device_map,
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


def _set_property(model, key):
    if not hasattr(model, 'model'):
        return
    text_model = model.model
    if not hasattr(text_model, key) or hasattr(model.__class__, key):
        return

    def _value(self):
        return getattr(self.model, key)

    setattr(model.__class__, key, property(_value))


def fix_do_sample_warning(generation_config: GenerationConfig) -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.temperature == 0:
        generation_config.do_sample = False
    if generation_config.do_sample is False:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50


def get_model_list() -> List[str]:
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


class ModelLoader(BaseModelLoader):

    def __init__(
        self,
        model_info: ModelInfo,
        model_meta: ModelMeta,
        *,
        load_model: bool = False,
        # model kwargs
        attn_impl: Optional[str] = None,
        experts_impl: Optional[str] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_model_len: Optional[int] = None,
        auto_model_cls=None,
        return_dummy_model: bool = False,
        new_special_tokens: Optional[List[str]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_info = model_info
        self.model_meta = model_meta
        self.load_model = load_model
        attn_impl = attn_impl or kwargs.get('attn_implementation')
        self.attn_impl = attn_impl
        self.attn_impl_keys = None
        experts_impl = experts_impl or kwargs.get('experts_implementation')
        self.experts_impl = experts_impl
        self.rope_scaling = rope_scaling
        self.max_model_len = max_model_len
        self.auto_model_cls = auto_model_cls
        self.auto_config_cls = None
        self.auto_tokenizer_cls = None
        self.return_dummy_model = return_dummy_model
        self.new_special_tokens = new_special_tokens
        self.model_kwargs = model_kwargs
        self.patch_offload = kwargs.pop('patch_offload', False)
        self.init_strategy = kwargs.get('init_strategy')
        self.local_repo_path = kwargs.get('local_repo_path')
        self.leaf_modules = None
        self.pad_token = None
        if model_info.quant_method == 'fp8':
            self.torch_dtype = 'auto'
        else:
            self.torch_dtype = model_info.torch_dtype
        if version.parse(transformers.__version__) >= version.parse('4.56'):
            model_kwargs['dtype'] = self.torch_dtype
        else:
            model_kwargs['torch_dtype'] = self.torch_dtype
        _patch_awq_compat(model_info)
        logger.info(f'model_kwargs: {model_kwargs}')

    def _postprocess_config(self, config):
        # fix prediction_step (internvl2, ovis, ...)
        if not hasattr(config, 'keys_to_ignore_at_inference'):
            config.keys_to_ignore_at_inference = []
        if 'past_key_values' not in config.keys_to_ignore_at_inference:
            config.keys_to_ignore_at_inference.append('past_key_values')
        torch_dtype = self.model_info.torch_dtype
        HfConfigFactory.set_config_attr(config, 'torch_dtype', torch_dtype, include_vit=True)
        HfConfigFactory.compat_zero3(config)

        if self.rope_scaling:
            if transformers_5:
                rope_parameters = HfConfigFactory.get_config_attr(config, 'rope_parameters') or {}
                for key in ['rope_theta', 'partial_rotary_factor']:
                    if self.rope_scaling.get(key) is None and rope_parameters.get(key) is not None:
                        self.rope_scaling[key] = rope_parameters[key]
            HfConfigFactory.set_config_attr(config, 'rope_scaling', self.rope_scaling)
        if self.max_model_len:
            HfConfigFactory.set_max_model_len(config, self.max_model_len)
        num_labels = self.model_info.num_labels or getattr(config, 'num_labels', None)
        if num_labels and self.model_info.task_type in ['seq_cls', 'reranker']:
            self.model_info.num_labels = num_labels
            config.num_labels = num_labels
        problem_type = self.model_info.problem_type or getattr(config, 'problem_type', None)
        if problem_type and self.model_info.task_type == 'seq_cls':
            self.model_info.problem_type = problem_type
            config.problem_type = problem_type
        self._update_attn_impl(config)
        self.model_info.config = config
        return config

    def get_config(self, model_dir: str) -> PretrainedConfig:
        auto_config_cls = self.auto_config_cls or AutoConfig
        return auto_config_cls.from_pretrained(model_dir, trust_remote_code=True)

    def _get_tokenizer(self, processor):
        if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
            patch_getattr(processor.__class__, 'tokenizer')
        else:
            tokenizer = processor
        return tokenizer

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        auto_tokenizer_cls = self.auto_tokenizer_cls
        if auto_tokenizer_cls is None:
            if os.path.exists(os.path.join(model_dir, 'preprocessor_config.json')):
                from transformers import AutoProcessor
                auto_tokenizer_cls = AutoProcessor
            else:
                auto_tokenizer_cls = AutoTokenizer
        return auto_tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        if self.experts_impl is not None:
            model_kwargs['experts_implementation'] = self.experts_impl
        model_info = self.model_info
        model_meta = self.model_meta
        auto_model_cls = self.auto_model_cls
        model = None
        if model_info.task_type in {'seq_cls', 'reranker'} and auto_model_cls is None and not self.return_dummy_model:
            with patch_automodel_for_sequence_classification(model_config=config, patch_from_pretrained=False):
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir, config=config, trust_remote_code=True, **self.model_kwargs)
                    auto_model_cls = AutoModelForSequenceClassification
                except ValueError:
                    pass

        auto_model_cls = auto_model_cls or AutoModelForCausalLM
        context_kwargs = {
            'model_info': model_info,
            'model_meta': model_meta,
            'auto_model_cls': auto_model_cls,
            'return_dummy_model': self.return_dummy_model,
        }
        if model is None:
            if self.return_dummy_model:
                context = partial(patch_automodel, **context_kwargs)
            elif model_info.task_type == 'seq_cls' and not model_meta.is_reward:
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            elif model_info.task_type == 'seq_cls' and model_meta.is_reward and config.num_labels > 1:
                logger.warning('You are using a reward model for seq_cls task and num_labels > 1, '
                               'ignore_mismatched_sizes will be set to True')
                model_kwargs['ignore_mismatched_sizes'] = True
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            elif model_info.task_type == 'reranker':
                # For reranker task, patch CausalLM to SequenceClassification with num_labels=1
                logger.info('Converting CausalLM to SequenceClassification for reranker task with num_labels=1')
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            else:
                context = partial(patch_automodel, **context_kwargs)
            with context():
                model = auto_model_cls.from_pretrained(model_dir, config=config, trust_remote_code=True, **model_kwargs)
        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(config, 'auto_map') and auto_model_cls.__name__ in config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = auto_model_cls.__name__

        if model_info.task_type == 'embedding' and auto_model_cls.__name__ != 'AutoModel':
            from swift.model.patcher import patch_output_normalizer
            patch_output_normalizer(model, model_meta=model_meta)
        elif model_info.task_type == 'generative_reranker':
            self._patch_generative_reranker(model, processor)
        if transformers_5:
            self._compat_transformers5(model)
        return model

    def _patch_generative_reranker(self, model, processor):
        tokenizer = self._get_tokenizer(processor)
        lm_head_model = get_lm_head_model(model, self.model_meta).lm_head

        def lm_head_forward(module, hidden_states):
            return get_generative_reranker_logits(module.weight, tokenizer, hidden_states)

        patch_module_forward(lm_head_model, lm_head_forward)

    def _postprocess_model(self, model_dir, model):
        model_info = self.model_info

        if self.init_strategy is not None:
            InitModelStrategy.init_parameters(model, self.init_strategy)
        # fix seq classification task
        if self.leaf_modules is not None or model_info.is_moe_model:
            # deepspeed zero3
            self._deepspeed_set_z3_leaf_modules(model, self.leaf_modules)
        model.model_info = self.model_info
        model.model_meta = self.model_meta
        model.model_dir = model_dir
        self._init_generation_config(model, model_dir)
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', self.pad_token)

    def _add_new_special_tokens(self, model, processor):
        if not self.new_special_tokens:
            return
        tokenizer = self._get_tokenizer(processor)
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': self.new_special_tokens})
        if num_new_tokens > 0:
            logger.info(f'Added {num_new_tokens} new special tokens.')

            if model is not None and not self.return_dummy_model:
                llm_model = get_lm_head_model(model, self.model_meta)
                origin_vocab_size = HfConfigFactory.get_config_attr(llm_model.config, 'vocab_size')
                if origin_vocab_size < len(tokenizer):
                    vocab_size = math.ceil(len(tokenizer) / 128) * 128
                    llm_model.resize_token_embeddings(vocab_size)
                    # fix transformers==4.52.4 qwen2.5-vl
                    HfConfigFactory.set_config_attr(llm_model.config, 'vocab_size', vocab_size)

    def _postprocess_processor(self, processor: Processor):
        tokenizer = self._get_tokenizer(processor)
        pad_token = tokenizer.pad_token_id
        if pad_token is None:
            pad_token = tokenizer.eos_token_id
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = pad_token
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None
        self.pad_token = pad_token
        tokenizer.model_info = self.model_info
        tokenizer.model_meta = self.model_meta

    def _compat_transformers5(self, model):
        if self.model_meta.is_multimodal:
            for key in ['language_model', 'vision_tower', 'multi_modal_projector', 'visual', 'vision_model']:
                _set_property(model, key)

    def _update_attn_impl(self, config):
        AttnImpl.update_attn_impl(config, self.attn_impl, self.attn_impl_keys)

    def _deepspeed_set_z3_leaf_modules(self, model, z3_leaf_modules):
        if not is_deepspeed_zero3_enabled():
            return
        try:
            hf_model_type = model.config.model_type
        except Exception:
            return
        if z3_leaf_modules is None:
            if hf_model_type == 'qwen3_vl_moe':
                from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
                z3_leaf_modules = [Qwen3VLMoeTextSparseMoeBlock]
            elif hf_model_type == 'qwen3_omni_moe':
                from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import \
                    Qwen3OmniMoeThinkerTextSparseMoeBlock
                z3_leaf_modules = [Qwen3OmniMoeThinkerTextSparseMoeBlock]
            elif hf_model_type == 'qwen2_moe':
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
                z3_leaf_modules = [Qwen2MoeSparseMoeBlock]
            elif hf_model_type == 'qwen3_moe':
                from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
                z3_leaf_modules = [Qwen3MoeSparseMoeBlock]
            elif hf_model_type == 'glm4_moe':
                from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
                z3_leaf_modules = [Glm4MoeMoE]
            elif hf_model_type == 'glm4_moe_lite':
                from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteMoE
                z3_leaf_modules = [Glm4MoeLiteMoE]
            elif hf_model_type == 'glm4v_moe':
                from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextMoE
                z3_leaf_modules = [Glm4vMoeTextMoE]
            elif hf_model_type == 'gpt_oss':
                from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
                z3_leaf_modules = [GptOssMLP]
            elif hf_model_type == 'llama4':
                from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
                z3_leaf_modules = [Llama4TextMoe]
            elif hf_model_type == 'qwen3_next':
                from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
                z3_leaf_modules = [Qwen3NextSparseMoeBlock]
            elif hf_model_type == 'olmoe':
                from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
                z3_leaf_modules = [OlmoeSparseMoeBlock]

        if z3_leaf_modules:
            from deepspeed.utils import set_z3_leaf_modules
            set_z3_leaf_modules(model, z3_leaf_modules)
            logger.info(f'Setting z3_leaf_modules: {z3_leaf_modules}')

    def _init_generation_config(self, model, model_dir):
        # generation_config
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        if not hasattr(model, 'generation_config') and os.path.isfile(generation_config_path):
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        # fix llama2 warning
        if getattr(model, 'generation_config', None):
            fix_do_sample_warning(model.generation_config)

    def _get_model_processor(self, model_dir, config):
        processor = self.get_processor(model_dir, config)
        model = None
        if self.load_model:
            model = self.get_model(model_dir, config, processor, self.model_kwargs.copy())
        return model, processor

    def load(self) -> Tuple[Optional[PreTrainedModel], Processor]:
        patch_offload_context = patch_attach_align_device_hook_on_blocks() if self.patch_offload else nullcontext()
        model_dir = self.model_info.model_dir
        with patch_get_dynamic_module(), patch_tp_plan(self.load_model), patch_offload_context:
            config = self.get_config(model_dir)
            self._postprocess_config(config)
            model, processor = self._get_model_processor(model_dir, config)
            self._postprocess_processor(processor)
            if model:
                self._postprocess_model(model_dir, model)
        self._add_new_special_tokens(model, processor)
        return model, processor


class SentenceTransformersLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            model_dir, trust_remote_code=True, model_kwargs={
                'torch_dtype': self.torch_dtype,
            })
        model.config = config

        def enable_input_require_grads(self):

            def make_inputs_require_grads(module, input, output):
                output.requires_grad_(True)

            self._require_grads_hook = self[0].auto_model.embed_tokens.register_forward_hook(make_inputs_require_grads)

        model.enable_input_require_grads = MethodType(enable_input_require_grads, model)
        return model


class RewardModelLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        if 'AutoModel' in (getattr(config, 'auto_map', None) or {}):
            self.auto_model_cls = self.auto_model_cls or AutoModel
        return super().get_model(model_dir, config, processor, model_kwargs)


def get_model_processor(
    model_id_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Union[str, Dict[str, Any], None] = None,
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
    attn_impl: Optional[str] = None,
    experts_impl: Optional[str] = None,
    rope_scaling: Optional[Dict[str, Any]] = None,
    max_model_len: Optional[int] = None,
    auto_model_cls=None,
    new_special_tokens: Optional[List[str]] = None,
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker'] = None,
    num_labels: Optional[int] = None,
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None,
    return_dummy_model: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Optional[PreTrainedModel], Processor]:
    """Load a pretrained model and its processor from a model hub or local path.

    Args:
        model_id_or_path: The model identifier from a hub (HuggingFace/ModelScope) or local path.
        torch_dtype: Data type for model parameters. If None, uses the dtype from config.json.
        device_map: Device mapping strategy for model loading. If None, uses default device map.
            Can be a string (e.g., 'auto', 'cuda:0') or a dictionary mapping layers to devices.
        load_model: Whether to load the model weights. If False, only returns the processor.

        # Hub parameters
        use_hf: Force using HuggingFace Hub (True) or ModelScope (False). If None, it is controlled
            by the environment variable `USE_HF`, which defaults to '0'. Default: None.
        hub_token: Authentication token for accessing private models on the hub.
        revision: Specific model version to use.
        download_model: Whether to download model files. If None, determined by load_model value.

        # Model configuration
        model_type: Explicit model type when it cannot be uniquely determined from model_id_or_path/config.json.
        quantization_config: Configuration for model quantization.
        max_memory: Maximum memory allocation per device.
        attn_impl: Attention implementation. 'flash_attn' for Flash Attention, None for auto-select (sdpa/eager).
        experts_impl: experts implementation. Options are 'grouped_mm', 'batched_mm', 'eager'. Defaults to None.
            This feature requires "transformers>=5.0.0".
        rope_scaling: RoPE (Rotary Position Embedding) scaling configuration dictionary.
        max_model_len: Maximum sequence length the model can handle.
        auto_model_cls: Custom AutoModel class to use for loading (e.g., AutoModelForCausalLM).
        new_special_tokens: List of new special tokens to add to the tokenizer.
        task_type: Task type for the model. Options: 'causal_lm', 'seq_cls', 'embedding', 'reranker',
            'generative_reranker'.
        num_labels: Number of labels for classification tasks.
        problem_type: Type of classification problem: 'regression', 'single_label_classification',
            or 'multi_label_classification'.
        return_dummy_model: If True, returns a dummy model (without loading weights).
        model_kwargs: Additional keyword arguments passed to the model's from_pretrained method.
        **kwargs: Additional keyword arguments passed to the loader.

    Returns:
        A tuple of (model, processor) where:
            - model: The loaded PreTrainedModel instance, or None if load_model=False.
            - processor: The Processor (tokenizer, processor, etc.) for the model.

    Examples:
        >>> # Load model and processor with default settings
        >>> model, processor = get_model_processor('Qwen/Qwen2.5-7B-Instruct')

        >>> # Load only processor without model
        >>> _, processor = get_model_processor('Qwen/Qwen2.5-7B-Instruct', load_model=False)
    """
    if load_model:
        patch_mp_ddp()
    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model and not return_dummy_model
    model_info, model_meta = get_model_info_meta(
        model_id_or_path,
        torch_dtype=torch_dtype,
        use_hf=use_hf,
        hub_token=hub_token,
        revision=revision,
        download_model=download_model,
        model_type=model_type,
        quantization_config=quantization_config,
        task_type=task_type,
        num_labels=num_labels,
        problem_type=problem_type)
    if device_map is None:
        device_map = get_default_device_map()
    model_kwargs['device_map'] = device_map
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    if max_memory:
        model_kwargs['max_memory'] = max_memory
    loader = model_meta.loader(
        model_info,
        model_meta,
        load_model=load_model,
        attn_impl=attn_impl,
        experts_impl=experts_impl,
        rope_scaling=rope_scaling,
        max_model_len=max_model_len,
        auto_model_cls=auto_model_cls,
        return_dummy_model=return_dummy_model,
        new_special_tokens=new_special_tokens,
        model_kwargs=model_kwargs,
        **kwargs)
    return loader.load()


def get_processor(
    model_id_or_path: str,
    *,
    # hub
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    revision: Optional[str] = None,
    download_model: Optional[bool] = None,
    # model kwargs
    model_type: Optional[str] = None,
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker'] = None,
    num_labels: Optional[int] = None,
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None,
    **kwargs,
) -> Processor:
    """Load only the processor for a pretrained model.

        This is a convenience function that wraps `get_model_processor` with `load_model=False`,
        returning only the processor without loading the model weights.
    """
    return get_model_processor(
        model_id_or_path,
        use_hf=use_hf,
        hub_token=hub_token,
        revision=revision,
        download_model=download_model,
        model_type=model_type,
        task_type=task_type,
        num_labels=num_labels,
        problem_type=problem_type,
        load_model=False,
        **kwargs)[1]
