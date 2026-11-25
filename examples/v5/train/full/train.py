# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 FULL-PARAMETER SFT at the COMPONENT level -- the code twin of train.sh.
#
# Same algorithm and loop as ../lora/train.py (read its header for the per-line commentary); the
# ONLY difference is the tuner: tuner='full' unfreezes every weight instead of attaching a low-rank
# adapter. Consequences you can see in this file:
#   * TunerConfig(tuner='full') -- no lora_rank/target_modules; ALL parameters train;
#   * learning_rate drops 1e-4 -> 1e-5 (full weights move much more per step than an adapter);
#   * the checkpoint saves the whole model, not an adapter (save_training_step handles both).
# Full-parameter training needs more GPU memory than LoRA (optimizer states for every weight);
# on models like this 4B one (large-memory) card is enough.
#
# RAY MODE: DistributedConfig(mode='ray', nproc_per_node=1) mirrors train.sh's single card.
# SIMPLIFIED vs the recipe: no resume / periodic eval / best-model tracking / perf profiling --
# the full versions live in swift/dev/recipe/supervised.py::run_supervised (what train.sh runs).
#
# Run: CUDA_VISIBLE_DEVICES=<free card> /usr/local/bin/python examples/v5/train/full/train.py
# Convergence signal: the loss falling (self-cognition is new knowledge for the model).

import math

from swift.dev.builders.optimizer import resolve_max_grad_norm
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
from swift.dev.loss import configure_loss
from swift.dev.recipe.sft.assembly import build_supervised_objects
from swift.dev.recipe.utils import (
    StepState,
    build_save_knobs,
    grouped_training_batches,
    save_final_checkpoint,
    save_training_step,
    snapshot_initial_adapter,
    write_ckpt_args_json,
)
# MEASURED: not run — FAIL: tuner='full' not supported in component API (2026-11-24)


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as train.sh."""
    return {
        'model_config': ModelConfig(attn_impl='flash_attention_2', model='Qwen/Qwen3.5-4B', task_type='causal_lm',
                                    torch_dtype='bfloat16'),
        'template_config': TemplateConfig(max_length=2048),
        'dataset_config': DatasetConfig(dataset=['swift/self-cognition#600'], split_dataset_ratio=0.01),
        'train_config': TrainConfig(
            num_train_epochs=3, per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-5, lr_scheduler='cosine', warmup_ratio=0.05),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=1),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=5),
        # FULL-PARAMETER: every weight trains; the lr above is the full-param-scale knob.
        'tuner_config': TunerConfig(tuner='full'),
        'quantize_config': QuantizeConfig(),
    }


def main() -> None:
    cfg = build_configs()
    model_config, template_config = cfg['model_config'], cfg['template_config']
    dataset_config, train_config = cfg['dataset_config'], cfg['train_config']
    distributed_config, checkpoint_config = cfg['distributed_config'], cfg['checkpoint_config']
    tuner_config, logging_config = cfg['tuner_config'], cfg['logging_config']
    quantize_config = cfg['quantize_config']
    output_dir = checkpoint_config.output_dir

    # THE OBJECTIVE: cross-entropy over the response tokens (plain SFT). Swap loss_type here to
    # train a different objective on the same loop (see run_sft.sft_loss for the full knob set).
    def sft_loss(model):
        configure_loss(model, loss_type=train_config.loss or 'cross_entropy')

    # ASSEMBLY: configs -> objects in the recipe's locked order (twinkle initialize -> model ->
    # tuner -> loss -> optimizer -> dataset/dataloader); validation runs inside.
    cluster = build_supervised_objects(
        recipe='run_sft', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, configure_loss=sft_loss, output_dir=output_dir)
    model, dataloader = cluster.model, cluster.dataloader
    ga = train_config.gradient_accumulation_steps

    # STEP CADENCE: counters + per-step metric record + periodic-save tail (shared StepState).
    state = StepState(
        model, train_config, gradient_accumulation_steps=ga, max_steps=cluster.total_opt_steps,
        max_grad_norm=resolve_max_grad_norm(train_config), logging_config=logging_config,
        output_dir=output_dir, save_steps=checkpoint_config.save_steps)
    save_knobs = build_save_knobs(checkpoint_config, dataset_config, distributed_config, tuner_config,
                                  output_dir=output_dir)

    def save(name: str, is_final: bool = False) -> str:
        state.publish_state()
        get_state = getattr(dataloader, 'get_state', None)
        dataloader_state = get_state() if get_state is not None else {}
        return save_training_step(
            model, state.handler, name, save_knobs,
            consumed_train_samples=int(dataloader_state.get('consumed_train_samples', 0)),
            component_states={'dataloader': dataloader_state}, is_final=is_final)

    # No-op for full-parameter (only base-modifying LoRA inits like pissa need the snapshot).
    snapshot_initial_adapter(model, save_knobs.initial_adapter_dir)
    state.publish_state()
    state.handler.on_train_begin()

    # THE TRAINING LOOP: one optimizer update per ga-batch group (see ../lora/train.py).
    epochs = math.ceil(train_config.num_train_epochs) if state.max_steps <= 0 else 10**9
    for epoch in range(epochs):
        if state.reached_max() or state.should_stop():
            break
        state.epoch = epoch
        if hasattr(dataloader, 'set_epoch'):
            dataloader.set_epoch(epoch)
        for group, batch_count in grouped_training_batches(dataloader, ga):
            state.micro_step += batch_count
            state.publish_state()
            state.handler.on_step_begin()
            # FORWARD + BACKWARD over the whole group; the backend splits per-rank micro-batches.
            model.forward_backward(
                inputs=group, task=cluster.forward_task, gradient_accumulation_steps=1,
                micro_batch_size=train_config.per_device_train_batch_size)
            state.handler.on_pre_optimizer_step()
            # OPTIMIZER STEP: gradient clip + update + LR schedule.
            model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
            state.handler.on_optimizer_step()
            state.record()
            state.step_tail(save=save)
            if state.reached_max() or state.should_stop():
                break

    state.handler.on_train_end()
    ckpt_dir = save_final_checkpoint(model, save, enabled=True)
    write_ckpt_args_json(ckpt_dir, cluster.processor, model_config, template_config, tuner_config,
                         quantize_config, dataset_config=dataset_config, task_type=cluster.task_type)
    state.finish_metrics(status='loop_completed')
    state.tracker.close()
    print(f'Full-parameter SFT finished: {state.global_step} optimizer steps; checkpoint at {ckpt_dir}')


if __name__ == '__main__':
    main()