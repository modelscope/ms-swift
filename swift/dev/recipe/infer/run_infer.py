from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# The generative and forward-pass bodies -- group sampling, the ``output_format`` emitters,
# the reward-channel wiring, the candidate cache / rollout recorder and the pooling /
# generative-reranker forwards -- live in
# :mod:`swift.dev.components.infer.component` (:class:`InferComponent`) now. The moved private
# names are re-exported here for backward compatibility, so the infer tests keep importing
# ``from swift.dev.recipe.infer.run_infer import X`` unchanged. (``_ArrowWriter`` has no definition
# anywhere -- a pre-existing broken name one test still imports -- and no move adds one.)
from swift.dev.components.infer.component import (
    _Candidate,
    _CandidateCache,
    _dpo_row,
    _emit_all,
    _emit_dpo,
    _is_too_easy,
    _plan_batches,
    _prompt_key,
    _sigmoid,
    compute_metric,
)
# The writer family moved one layer down since, to the storage layer's write side, which owns how a
# run streams into a store (:mod:`swift.dev.dataset.storage.writers`); the three names are
# re-exported from there for the same backward-compatibility reason.
from swift.dev.dataset.storage.writers import _CheckpointPaths, _CheckpointWriter, _IncrementalWriter

# The reward-channel layer -- resolving, building, placing and scoring a MODEL reward (``--orm`` / ``--prm``)
# -- lives in the shared ``swift.dev.rewards.scorer`` so online RL (grpo / ppo) drives the SAME code, not a
# second copy: a capability two businesses use belongs in a common file, and neither recipe imports the
# other. run_infer re-exports it under its historical private names purely for backward compatibility, so
# this module's call sites and the infer tests keep importing ``_resolve_reward_channel`` / ``_RewardSlot``
# / ... unchanged; new consumers import the public names from ``swift.dev.rewards.scorer`` directly.
from swift.dev.rewards.scorer import (
    SAMPLER_GROUP as _SAMPLER_GROUP,
    RewardModelSpec as _RewardModelSpec,
    RewardSlot as _RewardSlot,
    build_model_reward as _build_model_reward,
    exclusive_reward_groups as _exclusive_reward_groups,
    judge_query as _judge_query,
    normalize_scores as _normalize,
    parse_judge_score as _parse_judge_score,
    plan_sampling_device_groups,
    render_conversation as _render_conversation,
    resolve_reward_channel as _resolve_reward_channel,
    resolve_reward_model as _resolve_reward_model,
)
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TunerConfig,
    )

logger = get_logger()


def run_infer(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    infer_config: InferConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    rl_config: Optional[RLConfig] = None,
    rollout_config: Optional[RolloutConfig] = None,
    backend: str = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    distributed_config: Optional[DistributedConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    adapters: Optional[List[str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    plugin_config: Optional[PluginConfig] = None,
    merge_lora: bool = False,
    max_rows: Optional[int] = None,
    split_dataset_ratio: float = 0.01,
    output_path: Optional[str] = None,
    _shutdown: bool = True,
) -> List[Dict[str, Any]]:
    """Infer over a dataset: one pipeline, three task-type paths, and (for generation) three knobs.

    The generative path is a single pipeline -- sample a candidate group per prompt, optionally score
    it, then shape the rows -- whose behavior is set by three orthogonal knobs on ``infer_config``:
    ``num_return_sequences`` (candidates per prompt), the reward funcs (score every candidate, for any
    n), and ``output_format`` ('all' stores the whole group, 'dpo' stores best-of-n pairs, 'grpo' stores
    the whole scored group with group-relative advantages). A pooling
    ``task_type`` (seq_cls / embedding / reranker) or a ``generative_reranker`` instead runs a forward
    pass and returns one value per row.

    Args:
        model_config: model id/path, dtype, ``task_type`` (selects the path).
        template_config: chat template, and the ``system`` that overrides the dataset's.
        dataset_config: what to infer over. See ``split_dataset_ratio`` for how the split is chosen.
        infer_config: engine choice plus every generation/output/scoring knob (``sampler``,
            ``num_return_sequences``, ``output_format``, ``batch_size``, ``metric``, ``generation_strict``, the
            reward channels, resume / cache / rollout-token dumping).
        generation_config: decoding knobs. ``stream`` only affects the interactive REPL
            (:mod:`swift.dev.recipe.infer.infer_tui`); this dataset pipeline always samples in batch.
        rl_config: the reward-channel registry surface (``orm`` / ``prm`` and their ``*_weights``) and
            the multi-turn config (``max_turns`` / ``max_trajectory_tokens``), shared with GRPO.
        rollout_config: tool/sandbox config for a multi-turn rollout.
        backend: generation engine -- 'vllm' / 'sglang' / 'transformers' to sample locally, or 'no'
            for a cache-only run.
        engine_args: forwarded verbatim to the engine.
        distributed_config: DP / ray placement. The generative path gives the sampler -- and any
            GPU-resident reward model -- its own DeviceGroup; the forward paths use the 'model' group.
        tuner_config: source of ``adapters`` when not given explicitly.
        adapters: LoRA checkpoints. One adapter is selected per request (``adapter_path``); the engine
            is configured for LoRA at construction because vLLM cannot enable it later.
        merge_lora: fold the adapters into the base weights first and infer on the merged model.
        max_rows: stop after this many rows, for smoke tests.
        split_dataset_ratio: how much of ``dataset`` becomes the eval split when
            ``DatasetConfig.val_dataset`` is not set. 0 runs over the whole dataset.
        output_path: jsonl destination. None returns the rows without writing a file.
        _shutdown: leave True. False keeps the engine alive for tests that reuse it.

    Returns:
        Result rows. The generative 'all' path yields one row per prompt (``response`` / ``responses``
        / ``labels`` / ``messages``, plus ``scores`` when scored, plus every dataset column); 'dpo'
        yields chosen/rejected pairs; the forward paths yield one value per row.
    """
    from swift.dev.utils.plugin import PluginRegistry

    # A custom model / dataset / reward / tool lives in a plugin file, so its registration must precede
    # the first name lookup. ``load_configured`` declares swift's built-in kinds before importing the
    # user file (see ``_declare_builtin_kinds``), so an external ``@register('tool'/'reward', ...)`` finds
    # its kind already declared.
    PluginRegistry.load_configured(plugin_config)

    adapters = _resolve_adapters(adapters, tuner_config)
    if merge_lora and adapters:
        # run_merge_lora loads through transformers directly, so this needs no twinkle init and runs
        # before either path initializes.
        model_config, adapters = _merge_adapters(model_config, template_config, adapters)

    # One component serves every task type: InferComponent splits a sampling task (causal_lm) from a
    # forward-pass task (pooling / generative_reranker) inside its own __call__, so the recipe
    # constructs the node directly rather than through a per-task entry point.
    from swift.dev.components.infer.component import InferComponent

    return InferComponent(
        model_config,
        template_config,
        dataset_config,
        infer_config,
        generation_config,
        rl_config=rl_config,
        rollout_config=rollout_config,
        backend=backend,
        engine_args=engine_args,
        distributed_config=distributed_config,
        adapters=adapters,
        quantize_config=quantize_config,
        max_rows=max_rows,
        split_dataset_ratio=split_dataset_ratio,
        output_path=output_path,
        shutdown=_shutdown)()


def _resolve_adapters(adapters: Optional[List[str]], tuner_config: Optional[TunerConfig]) -> Optional[List[str]]:
    if adapters:
        return list(adapters)
    if tuner_config is not None and getattr(tuner_config, 'adapters', None):
        return list(tuner_config.adapters)
    return None


def _merge_adapters(model_config: ModelConfig, template_config: TemplateConfig,
                    adapters: List[str]) -> Tuple[ModelConfig, None]:
    """Merge the adapters into the base weights and return a ModelConfig pointing at the result.

    The returned config is a copy: mutating the caller's would make a second run with the same object
    silently infer on the merged directory. Adapters come back as None because they are now part of
    the weights -- passing them again would apply the same delta twice.
    """
    import dataclasses

    from swift.dev.config import TunerConfig
    from swift.dev.recipe.export.merge_lora import run_merge_lora

    merged = run_merge_lora(
        model_config, TunerConfig(adapters=list(adapters)), template_config=template_config, device_map='cpu')
    logger.info(f'run_infer: merged {len(adapters)} adapter(s) into {merged}')
    return dataclasses.replace(model_config, model=merged), None
