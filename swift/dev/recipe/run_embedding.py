from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        ModelConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

#: twinkle's task name; selects the embedding pooling patch inside forward.
TASK = 'embedding'


def run_embedding(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    tuner_config: Optional[TunerConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run embedding training from atomic Configs. Returns loss/grad_norm history.

    Orchestration mirrors ``run_sft`` step for step (see its docstring for the resume ordering
    contract, which is unchanged here); ``_save_final`` has the same test-oriented meaning.

    The loss is chosen by ``TrainConfig.loss`` (default 'infonce') and Matryoshka aggregation by
    ``TrainConfig.mrl_dims``; both are read here rather than passed separately so the same Config
    that drives legacy drives this recipe.
    """
    from swift.dev.loss import configure_embedding_loss, configure_embedding_metric
    from swift.dev.recipe.assembly import TrainAssembly
    from swift.dev.recipe.offline_loop import run_offline

    TrainAssembly.initialize_twinkle(
        distributed_config, seed=train_config.seed, full_determinism=train_config.full_determinism)

    def embedding_loss(model) -> None:
        # Unlike SFT, this runs on Megatron too: under task='embedding' the Megatron scheduler pools the
        # last stage's output and calls loss_instance explicitly instead of its internal vocab-parallel
        # CE, and its set_loss binds process_group to the DP group so InfonceLoss's in-batch all-gather
        # does not deadlock ranks on earlier pipeline stages.
        configure_embedding_loss(model, loss_type=train_config.loss or 'infonce', mrl_dims=train_config.mrl_dims)

    assembly = TrainAssembly(
        'run_embedding',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        task=TASK,
        output_dir=output_dir,
        logging_config=logging_config,
        quantize_config=quantize_config,
    )
    assembly.build_objects(embedding_loss, configure_metric=configure_embedding_metric)
    return run_offline(assembly, save_final=_save_final)
