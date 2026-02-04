# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Type, Union

import megatron.core
from megatron.core.models.gpt.gpt_layer_specs import (get_gpt_decoder_block_spec,
                                                      get_gpt_layer_with_transformer_engine_spec,
                                                      get_gpt_mtp_block_spec)
from packaging import version
from transformers.utils import is_torch_npu_available

from swift.model import MODEL_MAPPING
from swift.utils import get_logger
from .constant import MLLMMegatronModelType
from .model_config import create_mcore_model_config

if TYPE_CHECKING:
    from .gpt_model import GPTModel
    from .mm_gpt_model import MultimodalGPTModel

MEGATRON_MODEL_MAPPING = {}
logger = get_logger()
mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


@dataclass
class MegatronModelMeta:
    megatron_model_type: str
    model_types: List[str]

    loader: Optional[Type['MegatronModelLoader']] = None
    is_multimodal: bool = False

    # bridge_cls: Type[GPTBridge] = GPTBridge
    # model_cls: Optional[Type[nn.Module]] = None
    # get_transformer_layer_spec: Optional[Callable] = None
    # visual_cls: Optional[Type[nn.Module]] = None
    # get_mtp_block_spec: Optional[Callable] = None

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

    def __init__(self, args, hf_config):
        self.args = args
        self.hf_config = hf_config
        self.config = create_mcore_model_config(args, hf_config)
        self._check_npu()

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        args = self.args
        if self.config.num_experts:
            kwargs = {'qk_l2_norm': self.config.qk_l2_norm, 'vp_stage': vp_stage} if mcore_013 else {}
            # Define the decoder block spec
            transformer_layer_spec = get_gpt_decoder_block_spec(
                self.config, use_transformer_engine=True, normalization=self.config.normalization, **kwargs)
        else:
            transformer_layer_spec = self._get_transformer_layer_spec()

        if args.use_shared_expert_gate and args.num_experts and args.moe_shared_expert_intermediate_size:
            for layer_spec in transformer_layer_spec.layer_specs:
                if hasattr(layer_spec.submodules.mlp.submodules, 'shared_experts'):
                    layer_spec.submodules.mlp.submodules.shared_experts.params = {'gate': True}
        return transformer_layer_spec

    def _get_transformer_layer_spec(self):
        config = self.config
        kwargs = {'qk_l2_norm': config.qk_l2_norm} if mcore_013 else {}
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            config.num_experts,
            self.args.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            **kwargs,
        )
        return transformer_layer_spec

    def get_mtp_block_spec(self, transformer_layer_spec, vp_stage: Optional[int] = None):
        if self.args.mtp_num_layers is None:
            return
        if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            transformer_layer_spec_for_mtp = self._get_transformer_layer_spec()
        else:
            transformer_layer_spec_for_mtp = transformer_layer_spec
        kwargs = {'vp_stage': vp_stage} if mcore_013 else {}

        return get_gpt_mtp_block_spec(
            self.config, transformer_layer_spec_for_mtp, use_transformer_engine=True, **kwargs)

    def create_model_and_load(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ) -> Union[GPTModel, MultimodalGPTModel]:
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        mtp_block_spec = self.get_mtp_block_spec(transformer_layer_spec, vp_stage=vp_stage)
        return self._create_model(
            transformer_layer_spec,
            mtp_block_spec,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage)

    def _check_npu(self):
        MAX_NPU_EXPERTS_PER_EP = 128
        num_experts = self.config.num_experts
        if is_torch_npu_available() and num_experts > MAX_NPU_EXPERTS_PER_EP:
            required_ep = (num_experts + MAX_NPU_EXPERTS_PER_EP - 1) // MAX_NPU_EXPERTS_PER_EP
            if self.args.expert_model_parallel_size < required_ep:
                logger.warning(f'{">" * 20} WARNING {"<" * 20}\n'
                               f'MindSpeed on NPU supports up to {MAX_NPU_EXPERTS_PER_EP} experts per EP group. '
                               f'num_experts={num_experts}, '
                               f'expert_model_parallel_size={self.args.expert_model_parallel_size}. '
                               f'Please set expert_model_parallel_size (EP) to {required_ep} '
                               f'(num_experts / {MAX_NPU_EXPERTS_PER_EP}) or higher.')

    def _create_model(self,
                      transformer_layer_spec,
                      mtp_block_spec,
                      pre_process=True,
                      post_process=True,
                      vp_stage: Optional[int] = None):
        if self.args.is_multimodal:
            model_cls = MultimodalGPTModel
        else:
            model_cls = GPTModel
        return model_cls(
            config=self.config,
            transformer_layer_spec=transformer_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )
