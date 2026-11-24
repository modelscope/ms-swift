# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 KTO at the COMPONENT level -- the code twin of kto.sh, with the training process exposed.
#
# KTO (Kahneman-Tversky Optimization) learns from UNPAIRED feedback: each row is one response with
# a binary label (desirable / undesirable), not a chosen-vs-rejected pair. The loss pushes the
# policy's log-prob of desirable responses up and undesirable ones down, measured against a
# reference point z_KL -- a detached KL(pi||pi_ref) estimate the encoder computes over a
# MISMATCHED "KL batch" (each prompt paired with another row's completion). desirable_weight /
# undesirable_weight scale the two sides (class-imbalance knobs). A reference IS used (for z_KL
# and the per-row ref_logps), but there are no paired margins.
#
# What this file exposes: rl_type/beta/weights -> RLConfig; the KTO rotation + z_KL -> the
# PreferenceBatchEncoder seam; the update -> forward_backward + clip_grad_and_step in the loop.
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors kto.sh. SIMPLIFIED vs the recipe: no resume /
# best-model tracking -- full scaffolding in swift/dev/recipe/rlhf/run_dpo.py + run_supervised.
#
# Run: CUDA_VISIBLE_DEVICES=<2 free cards> /usr/local/bin/python examples/v5/rl/kto/kto.py
# Convergence signal: the loss falling (kto registers no algorithm metric -- bare loss/grad_norm;
# there are no rewards/margins tags for this type).

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
    """The run as typed Config objects -- the same knob values as kto.sh."""
    return {
        # attn_impl pinned to flash_attention_2: the default SDPA path lands on a cuDNN attention
        # kernel whose bf16 backward returns NaN on longer rows (measured on this box).
        'model_config': ModelConfig(model='Qwen/Qwen3.5-4B', torch_dtype='bfloat16',
                                    attn_impl='flash_attention_2'),
        'template_config': TemplateConfig(max_length=2048),
        # UNPAIRED feedback: each row carries one response + a binary desirable/undesirable label.
        'dataset_config': DatasetConfig(dataset=['AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#10000']),
        'train_config': TrainConfig(
            num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-4, warmup_ratio=0.05, gradient_checkpointing=True),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=5),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        # THE ALGORITHM KNOBS: beta = KL strength; desirable/undesirable_weight scale the two
        # feedback sides (raise desirable_weight when positive rows are scarce).
        'rl_config': RLConfig(rl_type='kto', beta=0.1, desirable_weight=1.0, undesirable_weight=1.0),
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

    # ASSEMBLY: like the SFT cluster, plus the preference pieces -- the KTO loss configured from
    # rl_config, and cluster.reference: the frozen initial policy the z_KL point is measured
    # against (with LoRA it is the policy itself with the adapter disabled).
    cluster = build_preference_objects(
        recipe='run_dpo', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rl_config=rl_config, reward_config=RewardConfig(),
        tuner_config=tuner_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=None, moe_config=None, output_dir=output_dir)
    model, dataloader = cluster.model, cluster.dataloader
    ga = train_config.gradient_accumulation_steps

    # THE KTO ENCODER (the family's micro_step seam): one call rotates the unpaired rows into
    # completion features + a mismatched KL batch, forwards the reference for ref_logps, and
    # estimates z_KL = mean(seq_logp_pi - seq_logp_ref).clamp(min=0) (detached). The KTO loss reads
    # ref_logps + z_kl off the forward kwargs. calculate_KL=False would anchor z_KL at 0.
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
            # FORWARD + BACKWARD under the KTO loss: per-row logistic surplus/deficit against the
            # z_KL reference point, weighted by the desirable/undesirable side.
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
    print(f'KTO finished: {state.global_step} optimizer steps; checkpoint at {ckpt_dir}')


if __name__ == '__main__':
    main()
