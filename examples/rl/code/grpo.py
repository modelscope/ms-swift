# Programmatic Python entry for v5 GRPO -- the code equivalent of ../grpo/grpo.sh.
#
# A .sh script spells the run as CLI flags; this file builds the SAME run as typed Config OBJECTS and drives
# it through swift's public programmatic API, so you can compute flags in Python (sweeps, conditional knobs,
# values from another pipeline) instead of hand-editing a shell string.
#
# The entry point is swift.dev.cli.rlhf.run_rlhf_configs(configs): it runs process_and_validate_configs then
# run_rlhf -- exactly what `swift rl` does after parsing argv. The ONE thing a programmatic caller must do
# itself is the PLACEMENT FORCING that normally lives in the argv parser (parse_rlhf_configs): every ONLINE
# rlhf_type (grpo/ppo/rft/gkd/opsd/mopd) rolls out on a separate vLLM sampler over Ray, so use_vllm/vllm_mode/
# mode='ray' are forced. _force_online_placement below mirrors it (parse_rlhf_configs is the source of truth).
#
# Run: python examples/rl/code/grpo.py   (set CUDA_VISIBLE_DEVICES to your free cards first).
import os

from swift.dev.cli.rlhf import run_rlhf_configs
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    PluginConfig,
    QuantizeConfig,
    RLHFConfig,
    RolloutConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)

#: Online rlhf_types roll out on a separate vLLM sampler over Ray (mirrors parse_rlhf_configs).
_ONLINE_TYPES = {'grpo', 'ppo', 'rft', 'gkd', 'opsd', 'mopd'}


def _force_online_placement(configs: dict) -> None:
    """Replicate the CLI's Ray+vLLM placement forcing, which run_rlhf_configs does NOT do.

    parse_rlhf_configs applies this after parsing argv; a programmatic caller bypasses that parser, so it
    must force the placement itself or the online loop would build with no sampler. Keep in sync with
    swift/dev/cli/rlhf.py::parse_rlhf_configs (the source of truth).
    """
    rlhf_config = configs['rlhf_config']
    if rlhf_config.rlhf_type in _ONLINE_TYPES:
        rollout_config = configs['rollout_config']
        rollout_config.use_vllm = True
        rollout_config.vllm_mode = rollout_config.vllm_mode or 'colocate'
        configs['distributed_config'].mode = 'ray'


def build_configs() -> dict:
    """Assemble the GRPO run as Config objects (the same knobs as ../grpo/grpo.sh)."""
    return {
        'model_config': ModelConfig(model='Qwen/Qwen2.5-7B-Instruct', torch_dtype='bfloat16'),
        'plugin_config': PluginConfig(),
        'template_config': TemplateConfig(max_length=3072),
        'dataset_config': DatasetConfig(dataset=['AI-MO/NuminaMath-TIR#5000']),
        'train_config': TrainConfig(
            learning_rate=1e-4,
            warmup_ratio=0.05,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
        ),
        # mode='ray' is forced by _force_online_placement; nproc_per_node IS required (in Ray mode the driver
        # orchestrates and holds no GPU). Pick free cards via CUDA_VISIBLE_DEVICES.
        'distributed_config': DistributedConfig(nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(
            output_dir='output', save_strategy='steps', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        'generation_config': GenerationConfig(temperature=1.0, max_new_tokens=2048),
        # GRPO is Ray-only with a weight-syncable vLLM sampler; colocate shares the trainer GPUs.
        'rollout_config': RolloutConfig(vllm_mode='colocate', vllm_gpu_memory_utilization=0.5),
        'rlhf_config': RLHFConfig(
            rlhf_type='grpo',
            orm=['accuracy'],
            num_generations=8,
            advantage_estimator='grpo',
            epsilon=0.2,
            num_iterations=1,
            beta=0.04,
            max_completion_length=2048,
            log_completions=True,
        ),
        'quantize_config': QuantizeConfig(),
        'megatron_config': None,
        'moe_config': None,
    }


def main() -> None:
    os.environ.setdefault('USE_SWIFT_V5', '1')
    configs = build_configs()
    _force_online_placement(configs)
    history = run_rlhf_configs(configs)
    # Judge convergence by the REWARD curve (train/total_reward climbing), not the loss -- see ../grpo/grpo.sh.
    print(f'GRPO finished: {len(history)} optimizer steps; '
          f'last reward={history[-1].get("reward") if history else None}')


if __name__ == '__main__':
    main()
