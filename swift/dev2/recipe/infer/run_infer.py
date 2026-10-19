# Copyright (c) ModelScope Contributors. All rights reserved.
# The infer entry point: plugin load, adapter resolution, optional merge-lora, then the component.
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch

if TYPE_CHECKING:
    from swift.dev2.config import (
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        QuantizeConfig,
        RewardConfig,
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
    reward_config: Optional[RewardConfig] = None,
    rollout_config: Optional[RolloutConfig] = None,
    backend: str = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    distributed_config: Optional[DistributedConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    adapters: Optional[List[str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    plugin_config: Optional[Any] = None,
    merge_lora: bool = False,
    max_rows: Optional[int] = None,
    split_dataset_ratio: float = 0.01,
    output_path: Optional[str] = None,
    _shutdown: bool = True,
) -> List[Dict[str, Any]]:
    """Infer over a dataset: plugin load, adapter resolution, then the InferComponent.

    The generative path -- sample a candidate group per prompt, optionally score it, then shape
    the rows -- is driven by ``num_return_sequences``, the reward funcs and ``output_format``
    ('all' the whole group, 'dpo' best-of-n pairs, 'grpo' the scored group). A pooling
    ``task_type`` or ``generative_reranker`` instead runs a forward pass, one value per row.
    """
    from swift.dev.utils.plugin import PluginRegistry

    # A plugin file's registration must precede the first name lookup: load_configured declares
    # swift's built-in kinds first, so an external @register finds its kind already declared.
    PluginRegistry.load_configured(plugin_config)

    adapters = _resolve_adapters(adapters, tuner_config)
    if merge_lora and adapters:
        # run_merge_lora loads through transformers directly, so this needs no twinkle init and
        # runs before either path initializes.
        model_config, adapters = _merge_adapters(model_config, template_config, adapters)

    # One component serves every task type: InferComponent splits the sampling task from the
    # forward-pass task inside its own __call__, so no per-task entry point is needed here.
    from swift.dev2.components.infer import InferComponent

    return InferComponent(
        model_config,
        template_config,
        dataset_config,
        infer_config,
        generation_config,
        reward_config=reward_config,
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

    The returned config is a copy: mutating the caller's would make a second run with the same
    object silently infer on the merged directory. Adapters come back as None because they are now
    part of the weights -- passing them again would apply the same delta twice.
    """
    import dataclasses

    from swift.dev2.config import TunerConfig
    from swift.dev2.recipe.export import run_merge_lora

    merged = run_merge_lora(
        model_config, TunerConfig(adapters=list(adapters)), template_config=template_config, device_map='cpu')
    logger.info(f'run_infer: merged {len(adapters)} adapter(s) into {merged}')
    return dataclasses.replace(model_config, model=merged), None
