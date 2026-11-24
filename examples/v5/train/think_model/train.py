# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 THINK-MODEL SFT at the COMPONENT level -- the code twin of train.sh.
#
# Qwen3-style hybrid reasoning models can answer directly OR think first (a think block before the
# answer). enable_thinking=True tells the template to render/supervise the thinking mode: the
# reasoning segment is part of the training target, so the model learns "think then answer".
# Set it False to train the non-thinking mode; unset follows the checkpoint default. Related knobs
# on TemplateConfig: preserve_thinking (keep existing think blocks in the data as-is) and
# add_non_thinking_prefix (mark non-thinking turns with an empty think prefix).
# Everything else is the standard LoRA SFT loop (see ../lora/train.py).
#
# RAY MODE: mode='ray', nproc_per_node=1 mirrors train.sh. SIMPLIFIED vs the recipe: no resume /
# eval / best-model tracking -- full scaffolding in swift/dev/recipe/supervised.py::run_supervised.
#
# Run: CUDA_VISIBLE_DEVICES=<free card> /usr/local/bin/python examples/v5/train/think_model/train.py
# Convergence signal: the loss falling (the model learns the supervised thinking behavior).

import math
from swift.dev.loss import configure_loss
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


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as train.sh."""
    return {
        'model_config': ModelConfig(model='Qwen/Qwen3-0.6B', task_type='causal_lm',
                                    torch_dtype='bfloat16'),
        # enable_thinking=True: supervise the think-then-answer mode; max_length roomier for the
        # extra reasoning tokens.
        'template_config': TemplateConfig(max_length=4096, enable_thinking=True),
        'dataset_config': DatasetConfig(dataset=['swift/self-cognition#600'], split_dataset_ratio=0.01),
        'train_config': TrainConfig(
            num_train_epochs=3, per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-4, lr_scheduler='cosine', warmup_ratio=0.05),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=1),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=5),
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

    # THE OBJECTIVE: cross-entropy over the response tokens (plain SFT). Swap loss_type here to
    # train a different objective on the same loop (see run_sft.sft_loss for the full knob set).
    def objective(model):
        configure_loss(model, loss_type=train_config.loss or 'cross_entropy')

    # ASSEMBLY: configs -> objects in the recipe's locked order (twinkle initialize -> model ->
    # tuner -> loss -> optimizer -> dataset/dataloader); validation runs inside.
    cluster = build_supervised_objects(
        recipe='run_sft', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, configure_loss=objective, output_dir=output_dir)
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
            # FORWARD + BACKWARD: the whole group is one call; the backend splits it into per-rank
            # micro-batches of per_device_train_batch_size and accumulates gradients across them.
            model.forward_backward(
                inputs=group, task=cluster.forward_task, gradient_accumulation_steps=1,
                micro_batch_size=train_config.per_device_train_batch_size)
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
    print(f'Think-model SFT finished: {state.global_step} optimizer steps; checkpoint at {ckpt_dir}')


if __name__ == '__main__':
    main()
