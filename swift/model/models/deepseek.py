# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from types import MethodType
from typing import Any, Dict

from swift.template import TemplateType
from swift.utils import Processor, get_logger, git_clone_github
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import ModelLoader, register_model
from ..utils import use_submodel_func


class DeepseekLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        # fix dtype bug
        mlp_cls = model.model.layers[-1].mlp.__class__

        for module in model.modules():
            if isinstance(module, mlp_cls):
                patch_output_to_input_device(module)
        return model


register_model(
    ModelMeta(
        LLMModelType.deepseek,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-moe-16b-chat', 'deepseek-ai/deepseek-moe-16b-chat'),
                Model('deepseek-ai/deepseek-moe-16b-base', 'deepseek-ai/deepseek-moe-16b-base'),
            ], ),
        ],
        DeepseekLoader,
        template=TemplateType.deepseek,
        architectures=['DeepseekForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_v2,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-Coder-V2-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Instruct'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Base', 'deepseek-ai/DeepSeek-Coder-V2-Base'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Base'),
                Model('deepseek-ai/DeepSeek-V2-Lite', 'deepseek-ai/DeepSeek-V2-Lite'),
                Model('deepseek-ai/DeepSeek-V2-Lite-Chat', 'deepseek-ai/DeepSeek-V2-Lite-Chat'),
                Model('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V2'),
                Model('deepseek-ai/DeepSeek-V2-Chat', 'deepseek-ai/DeepSeek-V2-Chat'),
            ], TemplateType.deepseek),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V2.5', 'deepseek-ai/DeepSeek-V2.5'),
                Model('deepseek-ai/DeepSeek-V2.5-1210', 'deepseek-ai/DeepSeek-V2.5-1210')
            ], TemplateType.deepseek_v2_5)
        ],
        DeepseekLoader,
        model_arch=ModelArch.deepseek_v2,
        architectures=['DeepseekV2ForCausalLM'],
        requires=['transformers>=4.39.3'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_v3,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V3-Base', 'deepseek-ai/DeepSeek-V3-Base'),
                Model('deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-V3'),
                Model('deepseek-ai/DeepSeek-V3-0324', 'deepseek-ai/DeepSeek-V3-0324'),
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('cognitivecomputations/DeepSeek-V3-awq', 'cognitivecomputations/DeepSeek-V3-AWQ'),
                Model('cognitivecomputations/DeepSeek-V3-0324-AWQ', 'cognitivecomputations/DeepSeek-V3-0324-AWQ')
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-Prover-V2-7B', 'deepseek-ai/DeepSeek-Prover-V2-7B'),
                Model('deepseek-ai/DeepSeek-Prover-V2-671B', 'deepseek-ai/DeepSeek-Prover-V2-671B'),
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('unsloth/DeepSeek-V3-bf16', 'unsloth/DeepSeek-V3-bf16'),
                Model('unsloth/DeepSeek-V3-0324-BF16', 'unsloth/DeepSeek-V3-0324-BF16'),
                Model('unsloth/DeepSeek-Prover-V2-671B-BF16', 'unsloth/DeepSeek-Prover-V2-671B-BF16'),
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-R1'),
                Model('deepseek-ai/DeepSeek-R1-Zero', 'deepseek-ai/DeepSeek-R1-Zero'),
                Model('deepseek-ai/DeepSeek-R1-0528', 'deepseek-ai/DeepSeek-R1-0528'),
            ], TemplateType.deepseek_r1),
            ModelGroup([
                Model('cognitivecomputations/DeepSeek-R1-awq', 'cognitivecomputations/DeepSeek-R1-AWQ'),
                Model('cognitivecomputations/DeepSeek-R1-0528-AWQ', 'cognitivecomputations/DeepSeek-R1-0528-AWQ'),
            ], TemplateType.deepseek_r1),
            ModelGroup([
                Model('unsloth/DeepSeek-R1-BF16', 'unsloth/DeepSeek-R1-BF16'),
                Model('unsloth/DeepSeek-R1-Zero-BF16', 'unsloth/DeepSeek-R1-Zero-BF16'),
                Model('unsloth/DeepSeek-R1-0528-BF16', 'unsloth/DeepSeek-R1-0528-BF16'),
            ], TemplateType.deepseek_r1),
            ModelGroup([
                Model('moonshotai/Moonlight-16B-A3B', 'moonshotai/Moonlight-16B-A3B'),
                Model('moonshotai/Moonlight-16B-A3B-Instruct', 'moonshotai/Moonlight-16B-A3B-Instruct'),
            ],
                       TemplateType.moonlight,
                       requires=['transformers<4.49']),
            ModelGroup([
                Model('moonshotai/Kimi-K2-Base', 'moonshotai/Kimi-K2-Base'),
                Model('moonshotai/Kimi-K2-Instruct', 'moonshotai/Kimi-K2-Instruct'),
                Model('moonshotai/Kimi-K2-Instruct-0905', 'moonshotai/Kimi-K2-Instruct-0905'),
                Model('moonshotai/Kimi-K2-Thinking', 'moonshotai/Kimi-K2-Thinking'),
            ], TemplateType.kimi_k2),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V3.1-Base', 'deepseek-ai/DeepSeek-V3.1-Base'),
                Model('deepseek-ai/DeepSeek-V3.1', 'deepseek-ai/DeepSeek-V3.1'),
                Model('deepseek-ai/DeepSeek-V3.1-Terminus', 'deepseek-ai/DeepSeek-V3.1-Terminus'),
            ], TemplateType.deepseek_v3_1),
        ],
        DeepseekLoader,
        model_arch=ModelArch.deepseek_v2,
        architectures=['DeepseekV3ForCausalLM'],
        requires=['transformers>=4.39.3'],
    ))


class DeepseekV32Loader(ModelLoader):

    def get_config(self, model_dir: str):
        try:
            from transformers.models.deepseek_v32 import DeepseekV32Config
        except ImportError:
            from transformers.models.deepseek_v3 import DeepseekV3Config as DeepseekV32Config
        return DeepseekV32Config.from_pretrained(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        try:
            from transformers.models.deepseek_v32 import DeepseekV32ForCausalLM
        except ImportError:
            # It’s only for compatibility with Megatron training or vllm/sglang infer,
            # while we wait for Transformers to support deepseek_v32.
            from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM as DeepseekV32ForCausalLM
            if not self.return_dummy_model:
                raise ValueError('DeepSeek-V3.2 is not supported in transformers.')
        self.auto_model_cls = DeepseekV32ForCausalLM
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.deepseek_v32,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V3.2', 'deepseek-ai/DeepSeek-V3.2'),
                Model('deepseek-ai/DeepSeek-V3.2-Speciale', 'deepseek-ai/DeepSeek-V3.2-Speciale'),
                Model('deepseek-ai/DeepSeek-V3.2-Exp', 'deepseek-ai/DeepSeek-V3.2-Exp'),
                Model('deepseek-ai/DeepSeek-V3.2-Exp-Base', 'deepseek-ai/DeepSeek-V3.2-Exp-Base'),
                Model('deepseek-ai/DeepSeek-Math-V2', 'deepseek-ai/DeepSeek-Math-V2'),
            ]),
        ],
        DeepseekV32Loader,
        template=TemplateType.deepseek_v3_1,
        architectures=['DeepseekV32ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_v4,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V4-Flash', 'deepseek-ai/DeepSeek-V4-Flash'),
                Model('deepseek-ai/DeepSeek-V4-Flash-Base', 'deepseek-ai/DeepSeek-V4-Flash-Base'),
            ]),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V4-Pro', 'deepseek-ai/DeepSeek-V4-Pro'),
                Model('deepseek-ai/DeepSeek-V4-Pro-Base', 'deepseek-ai/DeepSeek-V4-Pro-Base'),
            ]),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V4-Flash-0731', 'deepseek-ai/DeepSeek-V4-Flash-0731'),
                Model('deepseek-ai/DeepSeek-V4-Pro-0813', 'deepseek-ai/DeepSeek-V4-Pro-0813'),
            ],
                       template=TemplateType.deepseek_v4_flash),
        ],
        template=TemplateType.deepseek_v4,
        architectures=['DeepseekV4ForCausalLM'],
    ))


class DeepseekV41Loader(ModelLoader):
    # DeepSeek-V4.1 ships a composite `deepseek_v41` config (text + vision) whose
    # `model_type`s are unknown to transformers. Register lightweight config classes
    # so AutoConfig can load the on-disk config; the text config subclasses the
    # native DeepseekV4Config and only adds the CSA2 source-layer routing fields.
    def get_config(self, model_dir: str):
        import transformers.models.deepseek_v4.configuration_deepseek_v4 as _v4_config
        from huggingface_hub.dataclasses import strict
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        # CSA2 widens the per-layer ratio vocabulary beyond V4's {0, 4, 128}.
        _csa2_ratio_to_layer_type = {
            0: 'sliding_attention',
            1: 'compressed_sparse_attention',
            2: 'heavily_compressed_attention',
            4: 'compressed_sparse_attention',
            128: 'heavily_compressed_attention',
        }

        @strict
        class DeepseekV41TextConfig(DeepseekV4Config):
            model_type = 'deepseek_v41_text'
            default_num_hash_layers = 0  # V4.1 drops the Hash-MoE bootstrap
            # CSA2 source-layer routing
            kv_source_layer_ids: list | None = None
            index_source_layer_ids: list | None = None
            candidate_source_layer_id: int | None = None
            candidate_topk_blocks: int | None = None
            candidate_block_size: int | None = None

            def __post_init__(self, **kwargs):
                # super() consumes (pops) the legacy `compress_ratios` to derive
                # layer_types and discards it; peek it here so the raw per-layer
                # ratios survive for the megatron parser (CSA2 reads the int list).
                compress_ratios = kwargs.get('compress_ratios')
                old = dict(_v4_config._COMPRESS_RATIO_TO_LAYER_TYPE)
                _v4_config._COMPRESS_RATIO_TO_LAYER_TYPE.update(_csa2_ratio_to_layer_type)
                try:
                    super().__post_init__(**kwargs)
                finally:
                    _v4_config._COMPRESS_RATIO_TO_LAYER_TYPE.clear()
                    _v4_config._COMPRESS_RATIO_TO_LAYER_TYPE.update(old)
                self.compress_ratios = compress_ratios
                # Store for vLLM compat (V4 config.json carries `num_hash_layers`
                # but V4.1 derives it from `default_num_hash_layers=0`).
                if not hasattr(self, 'num_hash_layers'):
                    self.num_hash_layers = sum(1 for t in (self.mlp_layer_types or []) if t == 'hash_moe')

        class DeepseekV41VisionConfig(PretrainedConfig):
            model_type = 'deepseek_v41_vision'

        class DeepseekV41Config(PretrainedConfig):
            model_type = 'deepseek_v41'
            sub_configs = {'text_config': DeepseekV41TextConfig, 'vision_config': DeepseekV41VisionConfig}

            # mcore-bridge's _set_inv_freq reads config.rope_scaling and expects
            # the nested dict {'main': {…}, 'compress': {…}}.  The base class
            # property ``rope_scaling`` returns ``self.rope_parameters`` which,
            # via __getattr__ below, is intentionally flattened to the 'compress'
            # sub-dict for vLLM compat.  Override the property so mcore gets the
            # full nested dict while vLLM still gets the flat one via
            # ``config.rope_parameters``.
            @property
            def rope_scaling(self):
                return self.text_config.rope_parameters

            @rope_scaling.setter
            def rope_scaling(self, value):
                # Allow assignment (PretrainedConfig.__init__ may set it from
                # kwargs); silently ignore – the authoritative source is always
                # text_config.rope_parameters.
                pass

            def __init__(self, text_config=None, vision_config=None, image_token_id=None, **kwargs):
                if isinstance(text_config, dict):
                    text_config = DeepseekV41TextConfig(**text_config)
                elif text_config is None:
                    text_config = DeepseekV41TextConfig()
                if isinstance(vision_config, dict):
                    vision_config = DeepseekV41VisionConfig(**vision_config)
                self.text_config = text_config
                self.vision_config = vision_config
                self.image_token_id = image_token_id
                # The mcore backbone reads `hf_config.layer_types` off the composite config
                # (see mcore_bridge DSv4HybridSelfAttention); surface the text config's
                # CSA2-derived layer_types so the shared DSv4 attention can index it.
                self.layer_types = getattr(text_config, 'layer_types', None)
                # Version-compat shim. transformers >= 5.12 turns PreTrainedConfig into a
                # strict dataclass that runs the *generic* validate_layer_type, which only
                # accepts the global ('sparse', 'dense') MLP labels. DeepSeek-V4 overrides
                # that validator on its *text* config to accept its own 'moe'/'hash_moe'
                # vocabulary, but this composite subclasses PreTrainedConfig directly and
                # would otherwise fail validation against the legacy 'moe' labels it
                # delegates from text_config (via __getattr__). Surface an allowed-vocabulary
                # copy on the composite so the generic validator passes; text_config keeps
                # its own 'moe' labels for the native DeepSeek-V4 modeling path. Setting the
                # attribute explicitly (before super().__init__ runs the strict validation)
                # also stops __getattr__ from delegating the legacy labels. Overriding the
                # validator method is not enough: huggingface_hub's @strict captures the
                # validator functions at decoration time, so a subclass method override is
                # never invoked. Harmless on older transformers without the strict validator.
                _mlp_layer_types = getattr(text_config, 'mlp_layer_types', None)
                if _mlp_layer_types is not None:
                    _mlp_remap = {'moe': 'sparse', 'hash_moe': 'sparse'}
                    self.mlp_layer_types = [_mlp_remap.get(t, t) for t in _mlp_layer_types]
                super().__init__(**kwargs)

            def __getattr__(self, name):
                # Delegate unknown attribute lookups to text_config so that vLLM
                # and other consumers can read text-model attrs (vocab_size,
                # hidden_size, num_attention_heads, …) directly from the
                # composite config without needing explicit surfacing.
                try:
                    return super().__getattribute__(name)
                except AttributeError:
                    text_config = super().__getattribute__('text_config')
                    if hasattr(text_config, name):
                        value = getattr(text_config, name)
                        # vLLM's V4 rope builder expects a flat rope_parameters
                        # dict (keys like rope_type, rope_theta, …); the HF text
                        # config structures it as {'main': {…}, 'compress': {…}}.
                        # Return the 'compress' sub-dict which carries the full
                        # parameter set (incl. yarn scaling info);  the builder
                        # selects the right theta via per-layer compress_ratio.
                        if name == 'rope_parameters' and isinstance(value, dict):
                            compress = value.get('compress')
                            if isinstance(compress, dict):
                                return compress
                        return value
                    raise

        AutoConfig.register('deepseek_v41_text', DeepseekV41TextConfig, exist_ok=True)
        AutoConfig.register('deepseek_v41_vision', DeepseekV41VisionConfig, exist_ok=True)
        AutoConfig.register('deepseek_v41', DeepseekV41Config, exist_ok=True)

        # The megatron adapter save flow instantiates a dummy HF model via
        # ``AutoModelForCausalLM(config)`` to extract LoRA target modules.
        # Without an explicit model-class registration for the composite
        # ``DeepseekV41Config`` this lookup fails. We register a thin wrapper
        # that delegates to the native ``DeepseekV4ForCausalLM`` (text-only),
        # which is sufficient for enumerating named parameters on ``meta``.
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

        class DeepseekV41ForCausalLM(DeepseekV4ForCausalLM):
            config_class = DeepseekV41Config

            def __init__(self, config):
                # The parent expects a flat DeepseekV4Config; unwrap the composite.
                text_config = getattr(config, 'text_config', config)
                super().__init__(text_config)
                self.config = config

        AutoModelForCausalLM.register(DeepseekV41Config, DeepseekV41ForCausalLM, exist_ok=True)

        # Register in vLLM's model registry so the colocate rollout engine can
        # resolve the V4.1 architecture.  Internally V4.1 text layers are
        # structurally identical to V4, so we re-use the same vLLM impl.
        # We also monkey-patch load_weights to skip V4.1-only tensors
        # (bias_vl, engram, compressor, indexer) that the V4 vLLM model
        # doesn't instantiate.
        try:
            import importlib as _importlib
            # The external deep_gemm package may be outdated (missing
            # tf32_hc_prenorm_gemm, fp8_einsum, etc.).  Force-replace it
            # with vLLM's bundled copy which ships a complete API set.
            import sys as _sys
            from vllm.model_executor.models.registry import ModelRegistry
            from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM as _V4CausalLM
            try:
                _bundled_dg = _importlib.import_module('vllm.third_party.deep_gemm')
                _sys.modules['deep_gemm'] = _bundled_dg
            except ImportError:
                pass

            if 'DeepseekV41ForCausalLM' not in ModelRegistry.get_supported_archs():
                # Monkey-patch _o_proj: V4's FlashMLA always uses FP8
                # einsum for the output projection.  V4.1 bf16 checkpoints
                # don't carry quantisation scales, so replace it with a
                # simple bf16 grouped matmul (skips inverse RoPE too —
                # acceptable for the tiny smoke-test).
                from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention as _FlashMLA

                def _bf16_o_proj(self, o, positions):
                    import torch as _torch
                    B = o.shape[0]
                    hpg = self.n_local_heads // self.n_local_groups
                    # (B, G, hpg*head_dim)
                    o_g = o.reshape(B, self.n_local_groups, hpg * (self.nope_head_dim + self.rope_head_dim))
                    w = self.wo_a.weight.data.view(self.n_local_groups, self.o_lora_rank, -1)
                    z = _torch.einsum('bgr,gdr->bgd', o_g.float(), w.float()).to(o.dtype)
                    result = self.wo_b(z.reshape(B, -1))
                    return result[0] if isinstance(result, tuple) else result

                _FlashMLA._o_proj = _bf16_o_proj

                def _v41_compat_load_weights(self, weights):
                    import math as _math
                    from vllm.model_executor.models.utils import AutoWeightsLoader

                    # V4.1-specific weight prefixes that the V4 vLLM model
                    # doesn't have modules for:
                    loader = AutoWeightsLoader(
                        self,
                        skip_substrs=[
                            'mtp.',  # multi-token prediction
                            'bias_vl',  # visual language gate bias
                            'engram.',  # engram tables
                            'compressor.',  # CSA compressor
                            'indexer.',  # CSA indexer
                            'aligner.',  # vision-to-text aligner
                            'vision.',  # vision encoder
                            'image_start',  # vision special tokens
                            'image_end',
                            'image_newline',
                        ],
                    )
                    loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
                    self.model.finalize_mega_moe_weights()

                    # V4.1 doesn't ship hc_head_* weights; initialise them for
                    # identity-like behaviour (equal-weight stream average).
                    # sigmoid(base) = 1/hc_mult  →  base = log(p/(1-p))
                    if hasattr(self.model, 'hc_head_fn'):
                        hc_mult = self.model.hc_mult
                        p = 1.0 / hc_mult
                        base_val = _math.log(p / (1.0 - p))
                        self.model.hc_head_fn.data.zero_()
                        self.model.hc_head_base.data.fill_(base_val)
                        self.model.hc_head_scale.data.zero_()
                        loaded |= {
                            'model.hc_head_fn',
                            'model.hc_head_base',
                            'model.hc_head_scale',
                        }

                    # Dynamic fp8 quantisation names the scale ``weight_scale``
                    # but V4's _o_proj reads ``weight_scale_inv``. Alias it.
                    for _, mod in self.named_modules():
                        if hasattr(mod, 'weight_scale') and not hasattr(mod, 'weight_scale_inv'):
                            mod.weight_scale_inv = mod.weight_scale

                    return loaded

                _V4CausalLM.load_weights = _v41_compat_load_weights
                ModelRegistry.register_model(
                    'DeepseekV41ForCausalLM',
                    'vllm.models.deepseek_v4:DeepseekV4ForCausalLM',
                )
        except Exception:
            pass  # vLLM not installed or incompatible version

        return super().get_config(model_dir)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_v41,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V4.1-Flash', 'deepseek-ai/DeepSeek-V4.1-Flash'),
            ]),
        ],
        DeepseekV41Loader,
        template=TemplateType.deepseek_v41,
        architectures=['DeepseekV41ForCausalLM'],
        model_arch=ModelArch.deepseek_v41,
        tags=['vision'],
    ))


class DeepseekVLLoader(ModelLoader):

    def get_config(self, model_dir: str):
        # compat with python==3.10
        if sys.version_info.minor >= 10:
            import collections
            import collections.abc
            for type_name in collections.abc.__all__:
                setattr(collections, type_name, getattr(collections.abc, type_name))
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL')
        sys.path.append(local_repo_path)
        from deepseek_vl.models import VLChatProcessor
        self.auto_tokenizer_cls = VLChatProcessor
        return super().get_config(model_dir)

    def _get_model(self, model_dir: str, llm_prefix, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        llm = getattr(model, llm_prefix)
        patch_output_clone(llm.model.embed_tokens)
        patch_output_to_input_device(llm.model.embed_tokens)
        use_submodel_func(model, llm_prefix)
        model.generation_config = llm.generation_config
        return model

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return self._get_model(model_dir, 'language_model', *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-vl-1.3b-chat', 'deepseek-ai/deepseek-vl-1.3b-chat'),
                Model('deepseek-ai/deepseek-vl-7b-chat', 'deepseek-ai/deepseek-vl-7b-chat'),
            ], ),
        ],
        DeepseekVLLoader,
        template=TemplateType.deepseek_vl,
        architectures=['MultiModalityCausalLM'],
        model_arch=ModelArch.deepseek_vl,
        tags=['vision'],
    ))


class DeepseekJanusLoader(DeepseekVLLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return self._get_model(model_dir, 'language_model', *args, **kwargs)

    def get_config(self, model_dir: str):
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/deepseek-ai/Janus')
        sys.path.append(local_repo_path)
        from janus.models import VLChatProcessor
        self.auto_tokenizer_cls = VLChatProcessor
        return super(DeepseekVLLoader, self).get_config(model_dir)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_janus,
        [
            ModelGroup([
                Model('deepseek-ai/Janus-1.3B', 'deepseek-ai/Janus-1.3B'),
            ]),
        ],
        DeepseekJanusLoader,
        template=TemplateType.deepseek_janus,
        model_arch=ModelArch.deepseek_janus,
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.deepseek_janus_pro,
        [
            ModelGroup([
                Model('deepseek-ai/Janus-Pro-1B', 'deepseek-ai/Janus-Pro-1B'),
                Model('deepseek-ai/Janus-Pro-7B', 'deepseek-ai/Janus-Pro-7B'),
            ]),
        ],
        DeepseekJanusLoader,
        template=TemplateType.deepseek_janus_pro,
        model_arch=ModelArch.deepseek_janus,
        tags=['vision'],
    ))


class DeepseekVL2Loader(DeepseekVLLoader):

    def get_config(self, model_dir: str):
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL2')
        sys.path.append(local_repo_path)
        try:
            from deepseek_vl2.models import DeepseekVLV2Processor
        except ImportError:
            # compat transformers>=4.42
            import transformers
            transformers.models.llama.modeling_llama.LlamaFlashAttention2 = None
            from deepseek_vl2.models import DeepseekVLV2Processor
        self.auto_tokenizer_cls = DeepseekVLV2Processor
        return super(DeepseekVLLoader, self).get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return super()._get_model(model_dir, 'language', *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl2,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-vl2-tiny', 'deepseek-ai/deepseek-vl2-tiny'),
                Model('deepseek-ai/deepseek-vl2-small', 'deepseek-ai/deepseek-vl2-small'),
                Model('deepseek-ai/deepseek-vl2', 'deepseek-ai/deepseek-vl2'),
            ]),
        ],
        DeepseekVL2Loader,
        template=TemplateType.deepseek_vl2,
        model_arch=ModelArch.deepseek_vl2,
        requires=['transformers<4.42'],
        tags=['vision'],
    ))


class DeepseekOCRLoader(ModelLoader):
    visual_name = 'vision_model'

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self.auto_model_cls = self.auto_model_cls or AutoModel
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.sam_model)
        patch_output_to_input_device(getattr(model.model, self.visual_name))
        patch_output_to_input_device(model.model.projector)
        return model

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        from transformers import AutoProcessor, AutoTokenizer

        # When not loading model (e.g., vllm backend), avoid triggering AutoConfig which would execute
        # trust_remote_code and cause transformers version compatibility issues
        # For vllm backend, we only need the processor/tokenizer
        try:
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:
            # Fallback to AutoTokenizer if AutoProcessor is not available
            processor = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        return processor


class DeepseekOCR2Loader(DeepseekOCRLoader):
    visual_name = 'qwen2_model'


register_model(
    ModelMeta(
        MLLMModelType.deepseek_ocr,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-OCR', 'deepseek-ai/DeepSeek-OCR'),
            ]),
        ],
        DeepseekOCRLoader,
        template=TemplateType.deepseek_ocr,
        model_arch=ModelArch.deepseek_ocr,
        architectures=['DeepseekOCRForCausalLM'],
        requires=['transformers==4.46.3', 'easydict'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.deepseek_ocr2,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-OCR-2', 'deepseek-ai/DeepSeek-OCR-2'),
            ]),
        ],
        DeepseekOCR2Loader,
        template=TemplateType.deepseek_ocr2,
        model_arch=ModelArch.deepseek_ocr2,
        architectures=['DeepseekOCR2ForCausalLM'],
        requires=['transformers==4.46.3', 'easydict'],
        tags=['vision'],
    ))


class UnlimitedOCRLoader(DeepseekOCRLoader):
    visual_name = 'vision_model'

    @staticmethod
    def _apply_multi_gpu_patch():
        modeling_module = None
        for mod_name, mod in sys.modules.items():
            if 'modeling_unlimitedocr' in mod_name:
                modeling_module = mod
                break

        if modeling_module is None:
            return False

        UnlimitedOCRModel = getattr(modeling_module, 'UnlimitedOCRModel', None)
        if UnlimitedOCRModel is None:
            return False

        if getattr(UnlimitedOCRModel, '_swift_multi_gpu_patched', False):
            return True

        _original_forward = UnlimitedOCRModel.forward

        def _patched_forward(self, *args, **kwargs):
            _orig_cat = torch.cat
            _orig_masked_scatter_ = torch.Tensor.masked_scatter_

            def _safe_cat(tensors, dim=0, **cat_kwargs):
                # Using the device of the first tensor as the reference, the others are aligned to it.
                ref_device = None
                for t in tensors:
                    if isinstance(t, torch.Tensor):
                        ref_device = t.device
                        break
                if ref_device is None:
                    return _orig_cat(tensors, dim, **cat_kwargs)
                aligned = [
                    t.to(ref_device) if isinstance(t, torch.Tensor) and t.device != ref_device else t for t in tensors
                ]
                return _orig_cat(aligned, dim, **cat_kwargs)

            def _safe_masked_scatter_(tensor_self, mask, source):
                # Use the device of tensor_self (inputs_embeds[idx]) as the reference.
                dev = tensor_self.device
                if mask.device != dev:
                    mask = mask.to(dev)
                if source.device != dev:
                    source = source.to(dev)
                return _orig_masked_scatter_(tensor_self, mask, source)

            modeling_module.torch.cat = _safe_cat
            torch.cat = _safe_cat
            torch.Tensor.masked_scatter_ = _safe_masked_scatter_
            try:
                return _original_forward(self, *args, **kwargs)
            finally:
                # Restore state
                modeling_module.torch.cat = _orig_cat
                torch.cat = _orig_cat
                torch.Tensor.masked_scatter_ = _orig_masked_scatter_

        UnlimitedOCRModel.forward = _patched_forward
        UnlimitedOCRModel._swift_multi_gpu_patched = True
        return True

    def get_model(self, model_dir: str, config, *args, **kwargs) -> PreTrainedModel:
        logger = get_logger()

        self.auto_model_cls = self.auto_model_cls or AutoModel

        def to_dict(self, *args, **kwargs):
            res = self._to_dict(*args, **kwargs)
            if 'language_config' in res and res['language_config'].get('torch_dtype') is not None:
                dtype = res['language_config']['torch_dtype']
                res['language_config']['torch_dtype'] = str(dtype).replace('torch.', '')
            res.pop('to_dict')
            res.pop('_to_dict')
            return res

        config._to_dict = config.to_dict
        config.to_dict = MethodType(to_dict, config)

        model = super(DeepseekOCRLoader, self).get_model(model_dir, config, *args, **kwargs)
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.sam_model)
        patch_output_to_input_device(getattr(model.model, self.visual_name))
        patch_output_to_input_device(model.model.projector)
        patch_output_to_input_device(model.model)

        _orig_sw = getattr(model.config, 'sliding_window_size', None)
        if _orig_sw is not None:
            model.config._ring_window = _orig_sw
            logger.info('[UnlimitedOCR] R-SWA enabled: ring_window=%d', _orig_sw)
            # Patch _prepare_4d_causal_attention_mask in the main process (where model.forward runs).
            # Without this, transformers mangles 4D R-SWA masks (0/-inf) by treating them as 0/1 binary.
            # Find DeepseekV2Model's module via MRO
            for cls in type(model.model).__mro__:
                if cls.__name__ == 'DeepseekV2Model':
                    mod = sys.modules[cls.__module__]
                    if not getattr(mod, '_rswa_patched_global', False):
                        _orig_fn = mod._prepare_4d_causal_attention_mask

                        def _passthrough_4d(attention_mask, *args, **kwargs):
                            if attention_mask is not None and attention_mask.ndim == 4:
                                return attention_mask
                            return _orig_fn(attention_mask, *args, **kwargs)

                        mod._prepare_4d_causal_attention_mask = _passthrough_4d
                        mod._rswa_patched_global = True
                        logger.info('[UnlimitedOCR] Patched _prepare_4d_causal_attention_mask in module %s',
                                    cls.__module__)
                    break
        else:
            logger.warning('[UnlimitedOCR] sliding_window_size config not found, R-SWA may not work.')

        # Fix device placement for bare nn.Parameter (image_newline, view_seperator)
        # These are used in torch.cat inside forward, so patch_output_to_input_device can't help.
        try:
            vision_device = next(model.model.vision_model.parameters()).device
            model.model.image_newline.data = model.model.image_newline.data.to(vision_device)
            model.model.view_seperator.data = model.model.view_seperator.data.to(vision_device)
        except Exception as e:
            logger.warning('[UnlimitedOCR] Failed to fix parameter device: %s', e)

        n_devices = len(set(str(p.device) for p in model.parameters() if p.device.type == 'cuda'))
        if n_devices > 1:
            if self._apply_multi_gpu_patch():
                logger.info('[UnlimitedOCR] Multi-GPU patch applied (%d GPUs).', n_devices)
            else:
                logger.warning('[UnlimitedOCR] Multi-GPU deployment failed to apply patch.'
                               'If an inference error occurs, please check whether'
                               ' `modeling_unlimitedocr` has been loaded correctly.')

        return model


register_model(
    ModelMeta(
        MLLMModelType.unlimited_ocr,
        [
            ModelGroup([
                Model('PaddlePaddle/Unlimited-OCR', 'PaddlePaddle/Unlimited-OCR'),
            ]),
        ],
        UnlimitedOCRLoader,
        template=TemplateType.unlimited_ocr,
        model_arch=ModelArch.unlimited_ocr,
        architectures=['UnlimitedOCRForCausalLM'],
        requires=['transformers==4.46.3', 'easydict'],
        tags=['vision'],
    ))
