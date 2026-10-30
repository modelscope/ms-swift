# Copyright (c) ModelScope Contributors. All rights reserved.
# The offline RLHF family: the run_dpo main plus the family's own training loop.
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

from swift.dev.recipe.supervised import run_supervised
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RewardConfig,
        RLConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()

#: rl_types this recipe handles (the online types have their own recipes).
_OFFLINE_TYPES = frozenset({"dpo", "kto", "cpo", "orpo", "simpo", "rm"})


def run_dpo(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: "CheckpointConfig",
    rl_config: "RLConfig",
    reward_config: Optional["RewardConfig"] = None,
    tuner_config: Optional["TunerConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    *,
    output_dir: str = "output",
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run an offline preference optimisation from atomic Configs; returns loss history.

    Dispatches on ``rl_config.rl_type`` (dpo/kto/cpo/orpo/simpo/rm): the preference data pipeline
    (raw rows -> chosen/rejected interleave, the KTO rotation + z_KL, the reference-logps forward)
    is the encoder's -- the family loop's ``micro_step_fn`` seam; every other stage rides
    :func:`build_preference_objects` and :func:`run_supervised`.
    """
    from swift.dev.recipe.rlhf.assembly import build_preference_objects
    from swift.dev.recipe.rlhf.encoder import PreferenceBatchEncoder, preference_log_line

    rl_type = rl_config.rl_type
    if rl_type not in _OFFLINE_TYPES:
        raise ValueError(
            f"run_dpo handles the offline preference types {sorted(_OFFLINE_TYPES)}, got "
            f"rl_type={rl_type!r}. Use run_grpo (online RL), run_distill (distillation) or "
            "run_ppo instead."
        )
    cluster = build_preference_objects(
        recipe="run_dpo",
        model_config=model_config,
        template_config=template_config,
        dataset_config=dataset_config,
        train_config=train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        rl_config=rl_config,
        reward_config=reward_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        output_dir=output_dir,
    )
    encoder = PreferenceBatchEncoder(
        cluster.model,
        cluster.template,
        rl_type=rl_type,
        reference=cluster.reference,
        calculate_kl=bool(rl_config.calculate_KL is not False),
        data_parallel_size=(cluster.model.device_mesh.data_world_size if distributed_config.mode == "ray" else 1),
    )
    return run_supervised(
        cluster,
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        quantize_config,
        megatron_config,
        micro_step_fn=encoder,
        output_dir=output_dir,
        _save_final=_save_final,
        log_formatter=preference_log_line,
    )
