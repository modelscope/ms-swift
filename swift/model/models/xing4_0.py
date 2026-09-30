# Copyright (c) ModelScope Contributors. All rights reserved.

import sys
import torch
import torch.nn as nn
import transformers
from packaging import version
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from typing import Optional

from swift.template import TemplateType
from swift.utils import get_env_args, get_logger
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()

transformers_5 = version.parse(transformers.__version__) >= version.parse('5.0.0.dev')

if transformers_5:
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
    from transformers.integrations.moe import use_experts_implementation


class Xing4_0Experts(nn.Module):
    """The routed experts of Xing4.0, with their weights stacked into 3D tensors.

    Same math as the official `Xing4_0MoE.moe`, which loops over 64 `Xing4_0MLP` modules in python.
    Stacking is what lets transformers>=5 dispatch to its grouped-GEMM expert backends, selected
    with `--experts_impl`.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        # Accumulate in the router dtype and cast back at the end, like the official loop does, so
        # that the eager path stays close to the grouped-GEMM one (which reduces in fp32 as well).
        final_hidden_states = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states)

        return final_hidden_states.to(hidden_states.dtype)


if transformers_5:
    # Decorated only under transformers>=5, where the interface exists. Below that the patch is
    # skipped entirely and the class is never instantiated.
    Xing4_0Experts = use_experts_implementation(Xing4_0Experts)


class Xing4_0Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self._patch_remote_code(model_dir, self.experts_impl)
        return super().get_model(model_dir, *args, **kwargs)

    @staticmethod
    def _patch_remote_code(model_dir: str, experts_impl: Optional[str] = None) -> None:
        """Speed up `modeling_xing4_0.py` in place instead of vendoring a copy of it.

        Two python loops dominate a decoder layer: the MoE walks its 64 experts one by one, and the
        mHC normalizes its combination matrix with 20 sinkhorn steps. Patching the module before
        `from_pretrained` resolves the classes keeps the loading path, and therefore the
        remote-code saving, unchanged.
        """
        model_cls = get_class_from_dynamic_module('modeling_xing4_0.Xing4_0ForCausalLM', model_dir)
        modeling_module = sys.modules[model_cls.__module__]
        if getattr(modeling_module, '_swift_patched', False):
            return
        modeling_module._swift_patched = True

        # Stacking only pays off with a grouped expert backend, so it follows `--experts_impl`:
        # off by default (keep the official per-expert nn.Linear, which all-linear LoRA covers and
        # which trains at grad parity), on automatically when a grouped backend is requested.
        if transformers_5 and get_env_args('swift_xing4_0_stacked_experts', bool, experts_impl is not None):
            Xing4_0Loader._stack_experts(modeling_module)
            # transformers gates the grouped_mm backend on a source-text heuristic: it greps the
            # model class's own module for the literal `@use_experts_implementation`. Our decorator
            # lives in this swift file, not in `modeling_xing4_0.py`, so the heuristic reads False
            # and an explicit `--experts_impl grouped_mm` raises. Set the cached flag it
            # short-circuits on (`_can_set_experts_implementation` returns early when it is a bool).
            # Set it on the shared base `Xing4_0PreTrainedModel`: both `Xing4_0ForCausalLM` and the
            # inner `Xing4_0Model` subclass it and each runs the check at instantiation, so the
            # attribute reaches both through the MRO.
            modeling_module.Xing4_0PreTrainedModel._can_set_experts_implementation_cached_value = True
        elif not transformers_5:
            logger.warning('The stacked-experts patch of Xing4.0 needs transformers>=5, skipped.')
        # Off by default: torch.compile fuses the mHC `collapsed` reduction and shifts its bf16
        # result by 1 ULP, which 38 layers of backward amplify into a ~6% (grouped) / ~33% (eager)
        # grad_norm deviation. Opt in for ~34% more end-to-end speed when that trade is acceptable.
        if get_env_args('swift_xing4_0_compile_mhc', bool, False):
            # Fuses the sinkhorn loop; Megatron-LM compiles the same function on its native path.
            modeling_module.Xing4_0HyperConnection.forward = torch.compile(
                modeling_module.Xing4_0HyperConnection.forward)

    @staticmethod
    def _stack_experts(modeling_module) -> None:
        # The checkpoint keeps the official `mlp.experts.{i}.{gate,up,down}_proj.weight` names. This
        # mapping merges them into the stacked tensors on load and splits them back inside
        # `save_pretrained`, so an exported model still loads with the unpatched remote code.
        register_checkpoint_conversion_mapping(
            'Xing4_0ForCausalLM', [
                WeightConverter(
                    source_patterns=['mlp.experts.*.gate_proj.weight', 'mlp.experts.*.up_proj.weight'],
                    target_patterns='mlp.experts.gate_up_proj',
                    operations=[MergeModulelist(dim=0), Concatenate(dim=1)]),
                WeightConverter(
                    source_patterns='mlp.experts.*.down_proj.weight',
                    target_patterns='mlp.experts.down_proj',
                    operations=[MergeModulelist(dim=0)]),
            ],
            overwrite=True)

        moe_cls = modeling_module.Xing4_0MoE

        def __init__(self, config):
            nn.Module.__init__(self)
            self.config = config
            self.experts = Xing4_0Experts(config)
            self.gate = modeling_module.Xing4_0TopkRouter(config)
            self.shared_experts = modeling_module.Xing4_0MLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)

        def moe(self, hidden_states, topk_indices, topk_weights):
            return self.experts(hidden_states, topk_indices, topk_weights)

        moe_cls.__init__ = __init__
        # `Xing4_0MoE.forward` calls `self.moe(...)`, so replacing that one method is enough.
        moe_cls.moe = moe


register_model(
    ModelMeta(
        LLMModelType.xing4_0,
        [
            ModelGroup([
                Model('XingChen-AGI/Xing4.0-29B-A4B', 'XingChen-AGI/Xing4.0-29B-A4B'),
            ]),
        ],
        Xing4_0Loader,
        template=TemplateType.xing4_0,
        # The MLA weight names are identical to DeepSeek-V2's, so its arch is reused.
        model_arch=ModelArch.deepseek_v2,
        architectures=['Xing4_0ForCausalLM'],
        requires=['transformers>=5.14'],
    ))
