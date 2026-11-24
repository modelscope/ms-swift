# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 ORPO at the COMPONENT level -- the code twin of orpo.sh, with the training process exposed.
#
# ORPO (Odds Ratio Preference Optimization) is a SINGLE-STAGE reference-free objective: the SFT CE
# on the chosen response plus an odds-ratio penalty that pushes the chosen/rejected odds apart --
# no reference model, no separate SFT stage. Here beta carries the odds-ratio term's weight
# (lambda_orpo in the paper). Like CPO/SimPO it skips the ref_logps forward entirely.
#
# What this file exposes: rl_type/beta -> RLConfig; the preference interleave -> the
# PreferenceBatchEncoder seam; the update -> forward_backward + clip_grad_and_step in the loop.
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors orpo.sh. SIMPLIFIED vs the recipe: no resume /
# best-model tracking -- full scaffolding in swift/dev/recipe/rlhf/run_dpo.py + run_supervised.
#
# Run: CUDA_VISIBLE_DEVICES=<2 free cards> /usr/local/bin/python examples/v5/rl/orpo/orpo.py
# Convergence signal: the loss falling; logps/chosen rising vs logps/rejected (NO rewards/* tags --
# reference-gated, and ORPO has no reference).

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
# MEASURED: peak VRAM ~3 GiB/card (Qwen3.5-0.8B, 4-step smoke, 1 card(s), 2026-11-24)


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as orpo.sh."""
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
        # THE ALGORITHM KNOBS: beta here is the odds-ratio penalty weight (lambda_orpo); the CE
        # term on the chosen response is always on (single-stage SFT+alignment).
        'rl_config': RLConfig(rl_type='orpo', beta=0.1),
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

    # ASSEMBLY: like the SFT cluster, plus the ORPO loss configured from rl_config. NO reference
    # model (cluster.reference is None) -- the odds-ratio term needs only the policy's own logps.
    cluster = build_preference_objects(
        recipe='run_dpo', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rl_config=rl_config, reward_config=RewardConfig(),
        tuner_config=tuner_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=None, moe_config=None, output_dir=output_dir)
    model, dataloader = cluster.model, cluster.dataloader
    ga = train_config.gradient_accumulation_steps

    # THE PREFERENCE ENCODER (the family's micro_step seam): one call interleaves chosen/rejected
    # into 2B sequences. ORPO is reference-free, so the encoder skips the ref_logps forward; the
    # loss computes the odds ratio from the policy logps alone.
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
            # FORWARD + BACKWARD under the ORPO loss: CE(chosen) + beta * odds-ratio penalty
            # (log odds(chosen) - log odds(rejected), sigmoid-shaped).
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
    print(f'ORPO finished: {state.global_step} optimizer steps; checkpoint at {ckpt_dir}')


if __name__ == '__main__':
    main()
