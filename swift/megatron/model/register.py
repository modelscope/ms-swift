# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Type, Union

import megatron.core
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (get_gpt_decoder_block_spec,
                                                      get_gpt_layer_with_transformer_engine_spec,
                                                      get_gpt_mtp_block_spec)
from packaging import version
from torch import nn

from swift.model import MODEL_MAPPING
from swift.utils import get_logger
from .constant import MLLMMegatronModelType
from .gpt_bridge import GPTBridge
from .model_config import get_mcore_model_config

if TYPE_CHECKING:
    from .gpt_model import GPTModel
    from .mm_gpt_model import MultimodalGPTModel

MEGATRON_MODEL_MAPPING = {}
logger = get_logger()


@dataclass
class MegatronModelMeta:
    megatron_model_type: str
    model_types: List[str]

    bridge_cls: Type[GPTBridge] = GPTBridge
    visual_cls: Optional[Type[nn.Module]] = None
    is_multimodal: bool = False
    loader: Optional[Type['MegatronModelLoader']] = None

    def __post_init__(self):
        if self.megatron_model_type in MLLMMegatronModelType.__dict__:
            self.is_multimodal = True
        if self.loader is None:
            self.loader = MegatronModelLoader


def register_megatron_model(megatron_model_meta: MegatronModelMeta, *, exist_ok: bool = False):
    megatron_model_type = megatron_model_meta.megatron_model_type
    for model_type in megatron_model_meta.model_types:
        model_meta = MODEL_MAPPING[model_type]
        model_meta.support_megatron = True
    if not exist_ok and megatron_model_type in MEGATRON_MODEL_MAPPING:
        raise ValueError(f'The `{megatron_model_type}` has already been registered in the MODEL_MAPPING.')
    MEGATRON_MODEL_MAPPING[megatron_model_type] = megatron_model_meta


_MODEL_META_MAPPING = None


def get_megatron_model_meta(model_type: str) -> Optional[MegatronModelMeta]:
    global _MODEL_META_MAPPING
    if _MODEL_META_MAPPING is None:
        _MODEL_META_MAPPING = {}
        for k, megatron_model_meta in MEGATRON_MODEL_MAPPING.items():
            for _model_type in megatron_model_meta.model_types:
                _MODEL_META_MAPPING[_model_type] = k
    if model_type not in _MODEL_META_MAPPING:
        return
    return MEGATRON_MODEL_MAPPING[_MODEL_META_MAPPING[model_type]]


class MegatronModelLoader:
    model_cls = None

    def __init__(self, args, hf_config):
        from swift.megatron.model import GPTModel, MultimodalGPTModel
        self.args = args
        self.hf_config = hf_config
        self.config = get_mcore_model_config(args, hf_config)
        self.mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        if self.model_cls is None:
            self.model_cls = MultimodalGPTModel if self.args.is_multimodal else GPTModel
        self._init_config()

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        if self.config.num_moe_experts:
            kwargs = {'qk_l2_norm': self.config.qk_l2_norm, 'vp_stage': vp_stage} if self.mcore_013 else {}
            transformer_layer_spec = get_gpt_decoder_block_spec(
                self.config, use_transformer_engine=True, normalization=self.config.normalization, **kwargs)
        else:
            transformer_layer_spec = self._get_transformer_layer_spec()
        return transformer_layer_spec

    def _get_transformer_layer_spec(self):
        config = self.config
        kwargs = {'qk_l2_norm': config.qk_l2_norm} if self.mcore_013 else {}
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            self.args.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            **kwargs,
        )
        return transformer_layer_spec

    def get_mtp_block_spec(self, transformer_layer_spec, vp_stage: Optional[int] = None):
        if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            # TODO: remove
            transformer_layer_spec_for_mtp = self._get_transformer_layer_spec()
        else:
            transformer_layer_spec_for_mtp = transformer_layer_spec
        kwargs = {'vp_stage': vp_stage} if self.mcore_013 else {}
        return get_gpt_mtp_block_spec(
            self.config, transformer_layer_spec_for_mtp, use_transformer_engine=True, **kwargs)

    def _set_shared_expert_gate(self, transformer_layer_spec):
        if (self.config.use_shared_expert_gate and self.config.num_moe_experts
                and self.config.moe_shared_expert_intermediate_size):
            for layer_spec in transformer_layer_spec.layer_specs:
                if hasattr(layer_spec.submodules.mlp.submodules, 'shared_experts'):
                    layer_spec.submodules.mlp.submodules.shared_experts.params = {'gate': True}

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ) -> Union['GPTModel', 'MultimodalGPTModel']:
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        self._set_shared_expert_gate(transformer_layer_spec)
        mtp_block_spec = None
        if self.args.mtp_num_layers is not None:
            mtp_block_spec = self.get_mtp_block_spec(transformer_layer_spec, vp_stage=vp_stage)
        model = self._init_model(
            transformer_layer_spec,
            mtp_block_spec,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage)
        return model

    def _init_config(self):
        config = self.config
        # apply_rope_fusion
        if config.apply_rope_fusion is not None:
            return
        if config.multi_latent_attention or config.rotary_interleaved:
            # Upgrading transformer_engine requires checking here.
            config.apply_rope_fusion = False
        else:
            config.apply_rope_fusion = True
        logger.info(f'Setting config.apply_rope_fusion: {config.apply_rope_fusion}.')

    def _init_model(self,
                    transformer_layer_spec,
                    mtp_block_spec,
                    pre_process=True,
                    post_process=True,
                    vp_stage: Optional[int] = None):
        return self.model_cls(
            config=self.config,
            transformer_layer_spec=transformer_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )


def get_mcore_model(args, hf_config):
    loader = args.megatron_model_meta.loader(args, hf_config)
    model_type = ModelType.encoder_or_decoder
    if (mpu.get_pipeline_model_parallel_world_size() > 1 and args.virtual_pipeline_model_parallel_size is not None):
        models = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
            post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
            model = loader.build_model(pre_process, post_process, vp_stage=i)
            model.model_type = model_type
            models.append(model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = loader.build_model(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type
        models = [model]
    return models
