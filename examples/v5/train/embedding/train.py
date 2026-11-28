# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: SMOKE-VERIFIED (bounded shrink: stsb:positive#64, max_steps=4; loss 1.18 -> 0.69,
# checkpoint saved; clean local Ray cluster).
# v5 EMBEDDING (InfoNCE) training at the COMPONENT level -- the code twin of train.sh.
#
# Trains a sentence-embedding model: each row is a (anchor, positive) pair; InfoNCE pulls the pair
# together against the OTHER rows' positives as in-batch negatives -- so per_device_train_batch_size
# IS the negative count, do not shrink it to 1. task='embedding' installs the pooling patch in
# forward; the whole group must be submitted as ONE forward (input_ids are built inside forward).
# attn_impl is pinned to flash_attention_2: the default SDPA path hits a cuDNN kernel whose bf16
# backward returns NaN on this box (measured). Convergence signal: pos_sim rising, neg_sim flat/down
# (EmbeddingMetric, registered via configure_metric) -- not the raw loss alone.
#
# RAY MODE: mode='ray', nproc_per_node=1 mirrors train.sh. SIMPLIFIED vs the recipe: no resume /
# eval / best-model tracking -- full scaffolding in swift/dev/recipe/supervised.py::run_supervised.
#
# Run: CUDA_VISIBLE_DEVICES=<free card> /usr/local/bin/python examples/v5/train/embedding/train.py

import math
from swift.dev.loss import configure_embedding_loss, configure_embedding_metric
from swift.dev.builder.optimizer import resolve_max_grad_norm
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
# MEASURED: peak VRAM ~3 GiB/card (embeddinggemma-300m, 4-step smoke, 1 card(s), 2026-11-24)


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as train.sh."""
    return {
        # attn_impl pinned: cuDNN SDPA bf16 backward NaNs on longer rows (see header).
        'model_config': ModelConfig(model='Qwen/Qwen3.5-4B', task_type='embedding',
                                    torch_dtype='bfloat16', attn_impl='flash_attention_2'),
        'template_config': TemplateConfig(max_length=512),
        # stsb positive pairs; batch 8 = 7 in-batch negatives per anchor.
        'dataset_config': DatasetConfig(dataset=['stsb:positive#2000']),
        'train_config': TrainConfig(
            num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=1,
            learning_rate=5e-5, lr_scheduler='cosine', warmup_ratio=0.05, loss='infonce'),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=1),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
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

    # THE OBJECTIVE: InfoNCE over in-batch negatives (positive pairs pulled together, other rows'
    # positives pushed apart). configure_embedding_loss also wires the Matryoshka dims (mrl_dims).
    # The embedding metric (pos_sim/neg_sim) registers via configure_metric AFTER the optimizer.
    def objective(model):
        configure_embedding_loss(model, loss_type=train_config.loss or 'infonce', mrl_dims=train_config.mrl_dims)

    # ASSEMBLY: configs -> objects in the recipe's locked order (twinkle initialize -> model ->
    # tuner -> loss -> optimizer -> dataset/dataloader); validation runs inside.
    cluster = build_supervised_objects(
        recipe='run_embedding', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, configure_loss=objective, output_dir=output_dir,
        task='embedding', configure_metric=configure_embedding_metric)
    model, dataloader = cluster.model, cluster.dataloader
    ga = train_config.gradient_accumulation_steps

    # STEP CADENCE: counters + per-step metric record (loss/grad_norm/lr) + periodic-save tail.
    state = StepState(
        model, train_config, gradient_accumulation_steps=ga, max_steps=cluster.total_opt_steps,
        max_grad_norm=resolve_max_grad_norm(train_config), logging_config=logging_config,
        output_dir=output_dir, save_steps=checkpoint_config.save_steps)
    save_knobs = build_save_knobs(checkpoint_config, dataset_config, distributed_config, tuner_config,
                                  output_dir=output_dir)

    def save(name: str, is_final: bool = False) -> str:
        state.publish_state()
        # Under ray the dataloader is a proxy: its position is only readable via the remote call.
        get_state = getattr(dataloader, 'get_state', None)
        dataloader_state = get_state() if get_state is not None else {}
        return save_training_step(
            model, state.handler, name, save_knobs,
            consumed_train_samples=int(dataloader_state.get('consumed_train_samples', 0)),
            component_states={'dataloader': dataloader_state}, is_final=is_final)

    # A base-modifying LoRA init (pissa...) needs its initial snapshot before the first step.
    snapshot_initial_adapter(model, save_knobs.initial_adapter_dir)
    state.publish_state()
    state.handler.on_train_begin()

    # THE TRAINING LOOP -- one optimizer update per yielded group of ga dataloader batches.
    epochs = math.ceil(train_config.num_train_epochs) if state.max_steps <= 0 else 10**9
    for epoch in range(epochs):
        if state.reached_max() or state.should_stop():
            break
        state.epoch = epoch
        # Reshuffle per epoch: twinkle's EpochSampler derives the order from the epoch number.
        if hasattr(dataloader, 'set_epoch'):
            dataloader.set_epoch(epoch)
        for group, batch_count in grouped_training_batches(dataloader, ga):
            state.micro_step += batch_count
            state.publish_state()
            state.handler.on_step_begin()
            # WHOLE-BATCH forward: embedding input_ids are produced INSIDE forward (the
            # InputProcessor builds them from the raw rows), so the sample-level micro-batcher
            # cannot split this task -- submit the whole group as one call (no micro_batch_size).
            model.forward_backward(inputs=group, task=cluster.forward_task, gradient_accumulation_steps=1)
            state.handler.on_pre_optimizer_step()
            # OPTIMIZER STEP: clip gradients to max_grad_norm, then apply the update + LR schedule.
            model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
            state.handler.on_optimizer_step()
            # Record one step (model.calculate_metric -> console + tensorboard) + periodic save.
            state.record()
            state.step_tail(save=save)
            if state.reached_max() or state.should_stop():
                break

    state.handler.on_train_end()
    # FINAL CHECKPOINT + args.json (model/template/tuner recorded so `swift infer --adapters` can
    # rebuild the run without flags).
    ckpt_dir = save_final_checkpoint(model, save, enabled=True)
    write_ckpt_args_json(ckpt_dir, cluster.processor, model_config, template_config, tuner_config,
                         quantize_config, dataset_config=dataset_config, task_type=cluster.task_type)
    state.finish_metrics(status='loop_completed')
    state.tracker.close()
    print(f'Embedding training finished: {state.global_step} optimizer steps; checkpoint at {ckpt_dir}')


if __name__ == '__main__':
    main()
