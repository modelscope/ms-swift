# Programmatic Python entry for v5 OPSD -- the code equivalent of ../opsd/opsd.sh.
#
# A .sh script spells the run as CLI flags; this file builds the SAME run as typed Config OBJECTS and drives
# it through swift's public programmatic API, so you can compute flags in Python (sweeps, conditional knobs,
# values from another pipeline) instead of hand-editing a shell string. See ./grpo.py for the full rationale.
#
# OPSD is an ONLINE distillation algorithm: the student rolls out on a vLLM sampler and a frozen teacher
# scores those completions, so it takes the SAME online placement forcing as GRPO (use_vllm + mode='ray').
# It also loads a custom dataset plugin (../opsd/opsd_plugin.py) via PluginConfig.external_plugins.
#
# Run: python examples/rl/code/opsd.py   (set CUDA_VISIBLE_DEVICES to your free cards first).
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

#: Online rl_types roll out on a separate vLLM sampler over Ray (mirrors parse_rl_configs).
_ONLINE_TYPES = {'grpo', 'ppo', 'rft', 'gkd', 'opsd', 'mopd'}

#: The OPSD dataset plugin, resolved from THIS file's location so the example runs from any CWD.
_OPSD_PLUGIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'opsd', 'opsd_plugin.py')


def _force_online_placement(configs: dict) -> None:
    """Replicate the CLI's Ray+vLLM placement forcing, which run_rl_configs does NOT do.

    parse_rl_configs applies this after parsing argv; a programmatic caller bypasses that parser, so it
    must force the placement itself or the online loop would build with no sampler. Keep in sync with
    swift/dev/cli/rl.py::parse_rl_configs (the source of truth).
    """
    rl_config = configs['rl_config']
    if rl_config.rl_type in _ONLINE_TYPES:
        rollout_config = configs['rollout_config']
        rollout_config.use_vllm = True
        rollout_config.vllm_mode = rollout_config.vllm_mode or 'colocate'
        configs['distributed_config'].mode = 'ray'


def build_configs() -> dict:
    """Assemble the OPSD run as Config objects (the same knobs as ../opsd/opsd.sh)."""
    return {
        'model_config': ModelConfig(model='Qwen/Qwen2.5-1.5B-Instruct', torch_dtype='bfloat16'),
        # external_plugins imports the dataset plugin BEFORE anything is built, so the `opsd_numina` dataset
        # name below resolves (PluginRegistry.load_configured; see ../opsd/opsd_plugin.py).
        'plugin_config': PluginConfig(external_plugins=[_OPSD_PLUGIN]),
        # max_length carries the PRIVILEGED teacher view (question + reference solution), which is longer than
        # the student prompt -- do not shrink it below the encoded reference length or rows are truncated.
        'template_config': TemplateConfig(max_length=8192),
        'dataset_config': DatasetConfig(dataset=['opsd_numina#5000']),
        'train_config': TrainConfig(
            learning_rate=2e-5,
            warmup_ratio=0.05,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
        ),
        'distributed_config': DistributedConfig(nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(
            output_dir='output', save_strategy='steps', save_steps=100, save_total_limit=2, save_only_model=True),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        'generation_config': GenerationConfig(max_new_tokens=2048),
        'rollout_config': RolloutConfig(vllm_mode='colocate', vllm_gpu_memory_utilization=0.5),
        # teacher_model is the frozen scorer; lmbda=1.0 is purely on-policy (required for the async/streaming
        # distillation loop); opsd_reverse selects the reverse-KL OPSD objective; num_generations=1 (a single
        # student rollout per prompt, teacher-scored).
        'rl_config': RLConfig(
            rl_type='opsd',
            teacher_model=['Qwen/Qwen2.5-7B-Instruct'],
            lmbda=1.0,
            opsd_reverse=True,
            num_generations=1,
            sampling_temperature=1.0,
            max_completion_length=2048,
        ),
        'quantize_config': QuantizeConfig(),
        'megatron_config': None,
        'moe_config': None,
    }


def main() -> None:
    os.environ.setdefault('USE_SWIFT_V5', '1')
    configs = build_configs()
    _force_online_placement(configs)
    history = run_rl_configs(configs)
    # OPSD's loss IS the teacher>student divergence: a FALLING loss is the convergence signal (no reward channel).
    print(f'OPSD finished: {len(history)} optimizer steps; last loss={history[-1].get("loss") if history else None}')


if __name__ == '__main__':
    main()
