# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.nn as nn
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from peft import LoraConfig


@dataclass(frozen=True)
class FSDPTurboModelSpec:
    model_type: str
    fsdp_modules: Tuple[str, ...]
    fsdp_hook_modules: Tuple[str, ...]
    tp_colwise_modules: Tuple[str, ...]
    tp_rowwise_modules: Tuple[str, ...]
    ep_modules: Tuple[str, ...]
    efsdp_modules: Tuple[str, ...]
    recompute_modules: Tuple[str, ...]
    module_patches: Tuple[Tuple[str, str], ...] = ()
    frozen_modules: Tuple[str, ...] = ()
    cp_function_patches: Tuple[Tuple[str, str], ...] = ()
    cp_module_patches: Tuple[Tuple[str, str], ...] = ()
    native_model_factory: Optional[str] = None
    eager_dispatcher: Optional[str] = None
    cp_implementation: Literal['ulysses', 'kv_allgather', 'kcp'] = 'ulysses'
    lora_modules: Tuple[str, ...] = ()

    def validate_lora_config(self, model: nn.Module, config: 'LoraConfig', *, tp_size: int) -> None:
        """Reject adapters whose targets bypass forward or conflict with TP."""
        from fsdp_turbo.utils.str_match import module_name_match
        from peft import LoraConfig
        from peft.tuners.tuners_utils import check_target_module_exists

        if not isinstance(config, LoraConfig) or (config.target_parameters or config.modules_to_save or config.use_dora
                                                  or config.lora_bias or config.bias != 'none'
                                                  or config.init_lora_weights not in (True, 'gaussian')):
            raise ValueError('FSDPTurbo requires vanilla linear PEFT LoRA, including when loading --adapters.')
        selected = {name for name, _ in model.named_modules() if check_target_module_exists(config, name)}
        if self.lora_modules:
            allowed = {name for names in self.matching_modules(model, self.lora_modules).values() for name in names}
        else:
            allowed = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        if not selected or not selected <= allowed:
            hint = ('Native V4.1 LoRA requires explicit attention targets; use --target_regex '
                    "'model\\.layers\\.\\d+\\.attn\\.(wq_a|wq_b|wkv|wo_b)'. "
                    if self.lora_modules else 'FSDPTurbo LoRA supports ordinary linear modules. ')
            raise ValueError(f'{hint}Unsupported targets: {sorted(selected - allowed)}')
        if tp_size > 1 and any(
                module_name_match(pattern, name) for name in selected
                for pattern in self.tp_colwise_modules + self.tp_rowwise_modules):
            raise ValueError('LoRA on TP-managed modules is not supported.')

    @staticmethod
    def matching_modules(model: nn.Module, patterns: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
        from fsdp_turbo.utils.str_match import module_name_match

        names = tuple(name for name, _ in model.named_modules())
        return {pattern: tuple(name for name in names if module_name_match(pattern, name)) for pattern in patterns}

    def validate_model(self,
                       model: nn.Module,
                       *,
                       require_tp: bool = False,
                       require_ep: bool = False,
                       require_recompute: bool = False) -> Dict[str, Dict[str, Tuple[str, ...]]]:
        groups = {
            'fsdp_modules': self.fsdp_modules,
            'fsdp_hook_modules': self.fsdp_hook_modules,
        }
        if require_tp:
            if not self.tp_colwise_modules or not self.tp_rowwise_modules:
                raise ValueError(f'FSDPTurbo TP is not implemented for {self.model_type}. Use tp_size=1.')
            groups.update({
                'tp_colwise_modules': self.tp_colwise_modules,
                'tp_rowwise_modules': self.tp_rowwise_modules,
            })
        if require_ep:
            groups.update({
                'ep_modules': self.ep_modules,
                'efsdp_modules': self.efsdp_modules,
            })
        if require_recompute:
            groups['recompute_modules'] = self.recompute_modules

        matches = {name: self.matching_modules(model, patterns) for name, patterns in groups.items()}
        missing = [
            f'{group}:{pattern}' for group, result in matches.items() for pattern, names in result.items() if not names
        ]
        if missing:
            raise RuntimeError(f'FSDPTurbo model spec did not match declared modules: {missing}.')
        return matches


_MODEL_SPECS = {
    'deepseek_v41':
    FSDPTurboModelSpec(
        model_type='deepseek_v41',
        fsdp_modules=('model.layers.{*}', 'model.embed', 'model.head', 'model.vision.blocks.{*}',
                      'model.vision.patch_embed', 'model.aligner'),
        fsdp_hook_modules=('model.layers.{*}', ),
        tp_colwise_modules=('model.layers.{*}.ffn.shared_experts.w1', 'model.layers.{*}.ffn.shared_experts.w3'),
        tp_rowwise_modules=('model.layers.{*}.ffn.shared_experts.w2', ),
        ep_modules=('model.layers.{*}.ffn.experts', ),
        efsdp_modules=('model.layers.{*}.ffn.experts', ),
        recompute_modules=('model.layers.{*}', ),
        frozen_modules=('model.layers.{*}.attn.indexer', ),
        native_model_factory='fsdp_turbo.models.deepseek.deepseek_v41.adapter.build_deepseek_v41_model',
        eager_dispatcher='custom_native_eager_forward',
        cp_implementation='kv_allgather',
        lora_modules=('model.layers.{*}.attn.wq_a', 'model.layers.{*}.attn.wq_b', 'model.layers.{*}.attn.wkv',
                      'model.layers.{*}.attn.wo_b'),
    ),
    'qwen3_5_moe':
    FSDPTurboModelSpec(
        model_type='qwen3_5_moe',
        fsdp_modules=(
            'model.visual.blocks.{*}',
            'model.visual.patch_embed',
            'model.visual.pos_embed',
            'model.language_model.layers.{*}',
            'model.language_model.embed_tokens',
            'lm_head',
        ),
        fsdp_hook_modules=('model.language_model.layers.{*}', ),
        tp_colwise_modules=('*.q_proj', '*.k_proj', '*.v_proj'),
        tp_rowwise_modules=('*.o_proj', ),
        ep_modules=('model.language_model.layers.{*}.mlp.experts', ),
        efsdp_modules=('model.language_model.layers.{*}.mlp.experts', ),
        recompute_modules=('model.language_model.layers.{*}', 'model.visual.blocks.{*}'),
        cp_function_patches=(
            ('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward', 'full_attention'),
            ('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet.forward', 'gated_delta_net'),
        ),
        cp_module_patches=(('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel.forward',
                            'fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5_moe_cp.qwen3_5_moe_model_forward'), ),
    ),
}


def get_model_spec(model_or_type: Union[nn.Module, str]) -> FSDPTurboModelSpec:
    if isinstance(model_or_type, str):
        model_type = model_or_type
    else:
        model_type = getattr(getattr(model_or_type, 'config', None), 'model_type', None)
    try:
        return _MODEL_SPECS[model_type]
    except KeyError as error:
        raise ValueError(f'No FSDPTurbo model spec is registered for model_type={model_type!r}.') from error
