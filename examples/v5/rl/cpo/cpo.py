# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 CPO at the COMPONENT level -- the code twin of cpo.sh, with the training process exposed.
#
# CPO (Contrastive Preference Optimization) is a REFERENCE-FREE approximation of DPO: it drops the
# frozen reference and the ref_logps forward, keeping the contrastive chosen-vs-rejected term and
# adding cpo_alpha * a supervised CE loss on the chosen response (the "behavior cloning" regularizer
# that keeps the policy fluent without a reference anchor). Cheaper per step than DPO (one fewer
# model forward); beta still scales the contrastive term.
#
# What this file exposes: rl_type/beta/cpo_alpha -> RLConfig; the preference interleave -> the
# PreferenceBatchEncoder seam (no reference forward for this type); the update -> the loop below.
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors cpo.sh. SIMPLIFIED vs the recipe: no resume /
# best-model tracking -- full scaffolding in swift/dev/recipe/rlhf/run_dpo.py + run_supervised.
#
# Run: CUDA_VISIBLE_DEVICES=<2 free cards> /usr/local/bin/python examples/v5/rl/cpo/cpo.py
# Convergence signal: the loss falling; logps/chosen rising vs logps/rejected (NO rewards/* tags --
# those are reference-gated and CPO has no reference).

import math

from swift.dev.builders.optimizer import resolve_max_grad_norm
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    QuantizeConfig,
    RewardConfig,
    RLConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)
from swift.dev.recipe.rlhf.assembly import build_preference_objects
from swift.dev.recipe.rlhf.encoder import PreferenceBatchEncoder
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
    """The run as typed Config objects -- the same knob values as cpo.sh."""
    return {
        # attn_impl pinned to flash_attention_2: the default SDPA path lands on a cuDNN attention
        # kernel whose bf16 backward returns NaN on longer rows (measured on this box).
        'model_config': ModelConfig(model='Qwen/Qwen3.5-4B', torch_dtype='bfloat16',
                                    attn_impl='flash_attention_2'),
        'template_config': TemplateConfig(max_length=2048),
        # A preference dataset: each row carries a prompt plus chosen/rejected responses.
        'dataset_config': DatasetConfig(dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000']),
        'train_config': TrainConfig(
            num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-4, warmup_ratio=0.05, gradient_checkpointing=True),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=5),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        # THE ALGORITHM KNOBS: beta scales the contrastive term; cpo_alpha weights the added
        # supervised CE on the chosen response (0 = pure contrastive, no BC regularizer).
        'rl_config': RLConfig(rl_type='cpo', beta=0.1, cpo_alpha=1.0),
        'quantize_config': QuantizeConfig(),
    }


def main() -> None:
    cfg = build_configs()
    model_config, template_config = cfg['model_config'], cfg['template_config']
    dataset_config, train_config = cfg['dataset_config'], cfg['train_config']
    distributed_config, checkpoint_config = cfg['distributed_config'], cfg['checkpoint_config']
    tuner_config, logging_config = cfg['tuner_config'], cfg['logging_config']
    rl_config, quantize_config = cfg['rl_config'], cfg['quantize_config']
    output_dir = checkpoint_config.output_dir

    # ASSEMBLY: like the SFT cluster, plus the CPO loss configured from rl_config. There is NO
    # reference model for this type (cluster.reference is None) -- that is what makes it cheaper.
    cluster = build_preference_objects(
        recipe='run_dpo', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rl_config=rl_config, reward_config=RewardConfig(),
        tuner_config=tuner_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=None, moe_config=None, output_dir=output_dir)
    model, dataloader = cluster.model, cluster.dataloader
    ga = train_config.gradient_accumulation_steps

    # THE PREFERENCE ENCODER (the family's micro_step seam): one call interleaves chosen/rejected
    # into 2B sequences. CPO is reference-free, so the encoder skips the ref_logps forward; the
    # loss reads the policy logps alone (plus the chosen CE term weighted by cpo_alpha).
    encoder = PreferenceBatchEncoder(
        model, cluster.template, rl_type=rl_config.rl_type, reference=cluster.reference,
        calculate_kl=bool(rl_config.calculate_KL is not False),
        data_parallel_size=model.device_mesh.data_world_size)

    # STEP CADENCE + CHECKPOINT components: identical to the SFT example (see train/lora/train.py
    # for the per-line commentary); the preference loop differs only in the encoder seam above.
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

    snapshot_initial_adapter(model, save_knobs.initial_adapter_dir)
    state.publish_state()
    state.handler.on_train_begin()

    # THE TRAINING LOOP: same cadence as SFT -- ga dataloader batches per optimizer update. The
    # record count of each group must divide the DP size (whole preference units per rank); the
    # encoder enforces that and doubles the micro-batch count for the interleaved sequences.
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
            # FORWARD + BACKWARD under the CPO loss: reference-free contrastive margin + cpo_alpha
            # * CE(chosen). No ref_logps ride the kwargs for this type.
            model.forward_backward(**encoder(group, micro_batch_size=train_config.per_device_train_batch_size))
            state.handler.on_pre_optimizer_step()
            # OPTIMIZER STEP: gradient clip + parameter update + LR schedule advance.
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
    print(f'CPO finished: {state.global_step} optimizer steps; checkpoint at {ckpt_dir}')


if __name__ == '__main__':
    main()
