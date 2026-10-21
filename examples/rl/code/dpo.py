# Programmatic Python entry for v5 DPO -- the code equivalent of ../dpo/dpo.sh.
#
# A .sh script spells the run as CLI flags; this file builds the SAME run as typed Config OBJECTS and drives
# it through swift's public programmatic API, so you can compute flags in Python (sweeps, conditional knobs,
# values from another pipeline) instead of hand-editing a shell string. See ./grpo.py for the full rationale.
#
# DPO is an OFFLINE preference algorithm: it forwards a fixed chosen/rejected dataset, it does NOT roll out a
# sampler. So the placement forcing differs from the online family -- only mode='ray' is forced (the policy is
# a Ray actor; a full-parameter run also puts a frozen reference on its own 'ref' group). No vLLM sampler.
#
# Run: python examples/rl/code/dpo.py   (set CUDA_VISIBLE_DEVICES to your free cards first).
import os

from swift.dev.cli.rl import run_rl_configs
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    PluginConfig,
    QuantizeConfig,
    RLConfig,
    RolloutConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)

#: Offline preference rl_types run mode='ray' but roll out nothing (mirrors parse_rl_configs).
_OFFLINE_TYPES = {'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'}


def _force_offline_placement(configs: dict) -> None:
    """Replicate the CLI's Ray forcing for the offline family, which run_rl_configs does NOT do.

    parse_rl_configs sets mode='ray' (and NO vLLM sampler) for dpo/kto/cpo/orpo/simpo/rm; a programmatic
    caller bypasses that parser, so it must force the mode itself. Keep in sync with
    swift/dev/cli/rl.py::parse_rl_configs (the source of truth).
    """
    if configs['rl_config'].rl_type in _OFFLINE_TYPES:
        configs['distributed_config'].mode = 'ray'


def build_configs() -> dict:
    """Assemble the DPO run as Config objects (the same knobs as ../dpo/dpo.sh)."""
    return {
        # attn_impl='flash_attention_2' is NOT cosmetic: the offline preference path's default SDPA picks a
        # cuDNN bf16 kernel whose backward returns NaN on this box, so DPO must pin flash_attention_2 (see
        # ../dpo/dpo.sh). A programmatic build has to set it just like the CLI script does.
        'model_config': ModelConfig(
            model='Qwen/Qwen2.5-7B-Instruct', torch_dtype='bfloat16', attn_impl='flash_attention_2'),
        'plugin_config': PluginConfig(),
        'template_config': TemplateConfig(max_length=2048),
        'dataset_config': DatasetConfig(dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000']),
        'train_config': TrainConfig(
            learning_rate=1e-4,
            warmup_ratio=0.05,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
        ),
        'distributed_config': DistributedConfig(nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(
            output_dir='output', save_strategy='steps', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=5),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        'generation_config': GenerationConfig(),
        # No sampler for an offline run; RolloutConfig stays at its inert defaults.
        'rollout_config': RolloutConfig(),
        # beta is the KL-to-reference strength; under LoRA the reference is the policy's own base with the
        # adapter disabled (no second model, no extra GPUs).
        'rl_config': RLConfig(rl_type='dpo', beta=0.1),
        'quantize_config': QuantizeConfig(),
        'megatron_config': None,
        'moe_config': None,
    }


def main() -> None:
    os.environ.setdefault('USE_SWIFT_V5', '1')
    configs = build_configs()
    _force_offline_placement(configs)
    history = run_rl_configs(configs)
    # DPO convergence: rewards/margins rising and rewards/accuracies -> 1 (a reference-gated DPOMetric tag).
    print(f'DPO finished: {len(history)} optimizer steps; last loss={history[-1].get("loss") if history else None}')


if __name__ == '__main__':
    main()
