# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 GKD at the COMPONENT level -- the code twin of gkd.sh, with the whole training process exposed.
#
# GKD (Generalized Knowledge Distillation): the STUDENT generates its own answers (on-policy
# rollouts) AND sometimes trains on the dataset's own completions (off-policy); a FROZEN 4B teacher
# scores the same response tokens with FULL-VOCAB LOGITS (signal='logits'), and the student matches
# them via a generalized Jensen-Shannon divergence -- beta interpolates the teacher/student mixture
# (beta=0.5 here). lmbda=0.5 is the ON-POLICY FRACTION: each training block flips a seeded coin --
# heads = the student's next rollout batch, tails = the next dataset-completion block (which also
# gets GKD's supervised CE term, via the worker's use_student=False flag). That coin draw is the
# algorithmic heart of GKD and is written out in the loop below.
#
# NOTE on temperature: gkd.sh passes --temperature 1.0, which lands on GenerationConfig; the distill
# rollout reads rl_config.sampling_temperature (falling back to distill_temperature), so this script
# sets sampling_temperature=1.0 explicitly -- the effective v5 knob for the rollout temperature.
#
# THE FIVE SEGMENTS mirror run_distill (shared with opsd/mopd; rl_type picks the signal):
#   1. prompts + the off-policy dataset rows (distill_dataset_rows encodes the completions);
#   2. devices (initialize_rl_devices + distill_group_requests for the frozen teacher);
#   3. cluster (initialize_distill_cluster: student, sampler, teacher, DistillWorker, engine);
#   4. driver (build_distill_driver: streams the student's rollouts, no reward scoring);
#   5. the loop (coin draw -> teacher logits signal -> forward_backward/clip_grad_and_step -> sync).
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors gkd.sh (colocate; the 4B teacher holds its own
# frozen group). SIMPLIFIED vs the recipe: no resume / perf profiling / manual-GC tuning; the
# lmbda draw, the teacher signal, the tail drop and the sync cadence are all kept.
#
# Run: CUDA_VISIBLE_DEVICES=<2 free cards> /usr/local/bin/python examples/v5/rl/gkd/gkd.py
# Convergence signal: the loss falling (it IS the teacher-student divergence; there is no reward).

import math

from swift.dev.builder import build_prompt_dataloader, resolve_rl_dp_size
from swift.dev.builder.optimizer import resolve_max_grad_norm
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    QuantizeConfig,
    RewardConfig,
    RLConfig,
    RolloutConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)
from swift.dev.recipe.batching import balance_update
from swift.dev.recipe.distill.assembly import (
    DISTILL_SIGNALS,
    build_distill_driver,
    distill_dataset_rows,
    distill_group_requests,
    initialize_distill_cluster,
)
from swift.dev.recipe.offload import sleep_for_rollout
from swift.dev.recipe.utils import (
    StepState,
    build_save_knobs,
    close_driver_preserving_error,
    effective_colocate,
    finalize_rl_resources,
    initialize_rl_devices,
    optimizer_batch_slices,
    prompt_rows_from_dataset,
    resolve_generation_batch_size,
    resolve_rollout_max_steps,
    rollout_step_budget,
    save_final_checkpoint,
    save_training_step,
    snapshot_initial_adapter,
    stream_batch_size,
)
from swift.dev.utils.scoring import copy_inputs
# MEASURED: not run — FAIL: FlashAttention fp32 vision bug (2026-11-24)


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as gkd.sh."""
    return {
        # The STUDENT (trained); the TEACHER lives on rl_config.teacher_model below.
        'model_config': ModelConfig(attn_impl='flash_attention_2', model='Qwen/Qwen3.5-4B', torch_dtype='bfloat16'),
        'template_config': TemplateConfig(max_length=8192),
        # alpaca rows carry assistant completions -- required for the lmbda<1 off-policy blocks.
        'dataset_config': DatasetConfig(dataset=['AI-ModelScope/alpaca-gpt4-data-zh#5000']),
        'train_config': TrainConfig(
            num_train_epochs=1, per_device_train_batch_size=4, gradient_accumulation_steps=1,
            learning_rate=2e-5, warmup_ratio=0.05, gradient_checkpointing=True),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2,
                                              save_only_model=True),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        # THE ALGORITHM KNOBS: teacher_model = the frozen logits scorer; lmbda = on-policy fraction
        # (the coin draw); beta = the JSD mixture interpolation; distill_temperature softens the
        # divergence; sampling_temperature = the rollout sampling heat (see header note).
        'rl_config': RLConfig(
            rl_type='gkd', teacher_model=['Qwen/Qwen3.5-4B'],
            lmbda=0.5, beta=0.5, num_generations=1, sampling_temperature=1.0,
            max_completion_length=2048),
        # THE ROLLOUT KNOBS: colocate shares the trainer GPUs with the sampler (sleep/wake
        # between phases); parameter_sync_step (default 1) publishes weights every N steps.
        'rollout_config': RolloutConfig(sampler_mode='colocate', rollout_sampler='vllm',
                                        vllm_gpu_memory_utilization=0.5),
        # Distillation scores no outcome channel -- the reward config stays empty.
        'reward_config': RewardConfig(),
        'quantize_config': QuantizeConfig(),
    }


def main() -> None:
    cfg = build_configs()
    model_config, template_config = cfg['model_config'], cfg['template_config']
    dataset_config, train_config = cfg['dataset_config'], cfg['train_config']
    distributed_config, checkpoint_config = cfg['distributed_config'], cfg['checkpoint_config']
    rollout_config, rl_config = cfg['rollout_config'], cfg['rl_config']
    reward_config, tuner_config = cfg['reward_config'], cfg['tuner_config']
    logging_config, quantize_config = cfg['logging_config'], cfg['quantize_config']
    output_dir = checkpoint_config.output_dir
    ga = train_config.gradient_accumulation_steps

    # SEGMENT 1 -- PROMPTS: read the raw rows; the resource estimator sizes device groups off the
    # prompt lengths, and the prompt dataloader streams these messages to the rollout driver.
    prompt_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)

    # SEGMENT 2 -- DEVICES: twinkle initialize (Ray), the template, and the device-group plan.
    # distill_group_requests declares the frozen teacher group(s) (a separate teacher id gets its
    # own GPUs; a disable-adapter/self teacher would reuse the policy and plan none).
    template, resource_plan, auxiliary_meshes = initialize_rl_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=None, moe_config=None,
        output_dir=output_dir, prompt_rows=prompt_rows,
        group_requests=(lambda: distill_group_requests(rl_config)))
    # The OFF-POLICY rows: dataset rows that carry their own assistant completion, encoded once
    # with the student's template (lmbda<1 trains some blocks on these instead of rollouts).
    dataset_rows = distill_dataset_rows(template, prompt_rows)
    if not dataset_rows:
        raise ValueError('GKD with lmbda<1 needs dataset rows with assistant completions.')


    # BATCH GEOMETRY: same derivation as run_distill -- one optimizer update trains
    # train_batch_size = per_device x dp rows; a rollout batch fills whole updates; the step budget
    # derives from prompt count x epochs. Distillation replays each batch once (num_iterations=1).
    dp_size = resolve_rl_dp_size(distributed_config, template_config.sequence_parallel_size)
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    generation_batch_size = resolve_generation_batch_size(
        rollout_config.generation_batch_size, num_prompts=len(prompts),
        num_generations=rl_config.num_generations, num_iterations=1,
        train_batch_size=train_batch_size, gradient_accumulation_steps=ga)
    colocate = effective_colocate(rollout_config, resource_plan)
    batch_size = stream_batch_size(
        colocate=colocate, generation_batch_size=generation_batch_size,
        num_generations=max(1, rl_config.num_generations), train_batch_size=train_batch_size,
        gradient_accumulation_steps=ga)
    num_train_epochs = math.ceil(train_config.num_train_epochs)
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size, gradient_accumulation_steps=ga,
        num_train_epochs=num_train_epochs, generation_batch_size=generation_batch_size,
        stream_batch_rows=batch_size)
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='gkd-example')

    # SEGMENT 3 -- THE CLUSTER: every actor distillation trains with, built explicitly. model = the
    # student policy (Ray actors, LoRA); sampler = the weight-syncable vLLM engine generating the
    # student's rollouts; teachers = the frozen scorer(s); worker = the DistillWorker that runs the
    # teacher forward per update and packages its signal as light training kwargs (the driver never
    # sees the teacher's tensors); engine = publishes student weights to the sampler.
    cluster = initialize_distill_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        tuner_config=tuner_config, max_steps=max_steps, signal=DISTILL_SIGNALS['gkd'],
        train_batch_size=train_batch_size, batch_size=batch_size,
        quantize_config=quantize_config, megatron_config=None, moe_config=None,
        generation_config=None, resource_plan=resource_plan, auxiliary_meshes=auxiliary_meshes)

    # SEGMENT 4 -- DATALOADER + DRIVER: the prompt stream feeds twinkle's StreamingDriver, which
    # generates the student's completions on the sampler and hands out whole batches. No ORM/PRM
    # scoring is wired -- the teacher signal is computed per update in the loop below.
    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=dataset_config.data_seed,
        shuffle=dataset_config.train_dataloader_shuffle, batch_size=generation_batch_size)
    steps_per_batch = max(1, batch_size // (train_batch_size * ga))
    driver = build_distill_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, prompt_dataloader=dataloader,
        batch_size=batch_size, num_train_epochs=num_train_epochs,
        max_steps=math.ceil(max_steps / steps_per_batch))

    # STEP CADENCE + CHECKPOINT: the recipe's shared StepState (counters, metric record, save tail).
    state = StepState(
        cluster.model, train_config, padding_free=template_config.padding_free,
        dp_balance=train_config.dp_balance, gradient_accumulation_steps=ga, max_steps=max_steps,
        max_grad_norm=resolve_max_grad_norm(train_config), logging_config=logging_config,
        output_dir=output_dir, save_steps=checkpoint_config.save_steps)
    save_knobs = build_save_knobs(checkpoint_config, dataset_config, distributed_config, tuner_config,
                                  output_dir=output_dir)

    def save(name: str, is_final: bool = False) -> str:
        state.publish_state()
        return save_training_step(cluster.model, state.handler, name, save_knobs,
                                  consumed_train_samples=state.global_step, is_final=is_final)

    # A base-modifying LoRA init (pissa...) needs its initial snapshot before the first step.
    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    # SEGMENT 5 -- THE VISIBLE LOOP with the lmbda COIN DRAW (the GKD mixing rule): every block
    # draws its source -- with probability lmbda the student's next driver batch (ON-policy), else
    # the next batch_size dataset completions (OFF-policy; the worker's use_student=False flag adds
    # GKD's supervised CE term on those rows). The draws before each on-policy block run first, so
    # the weight sync (which, colocated, hands the devices to the sampler) happens only right
    # before the driver is asked for a batch. This is run_distill._train_loop, written out.
    import random
    from itertools import count

    micro_batch_size = max(1, train_batch_size // dp_size)
    update_batch_size = train_batch_size * ga
    sync_every = rollout_config.parameter_sync_step
    draws = count()
    dataset_offset = 0

    def train_off_policy() -> bool:
        """Train dataset blocks until a block draws the student; False once training must stop."""
        nonlocal dataset_offset
        while not (state.reached_max() or state.should_stop()):
            if random.Random(train_config.seed + next(draws)).random() < rl_config.lmbda:
                return True
            rows = [dataset_rows[(dataset_offset + i) % len(dataset_rows)] for i in range(batch_size)]
            dataset_offset = (dataset_offset + batch_size) % len(dataset_rows)
            # OFF-POLICY UPDATE: dataset completions, lengths unknown (None), use_student=False.
            for lo, hi in optimizer_batch_slices(len(rows), train_batch_size, ga):
                if state.reached_max() or state.should_stop():
                    break
                update = rows[lo:hi]
                state.micro_step += (hi - lo) // train_batch_size
                update, balance_metrics = balance_update(
                    state, update, None,
                    micro_batch_kwargs={'micro_batch_size': micro_batch_size})
                # OFF-policy teacher signal: the same rows scored by the teacher; use_student=False
                # tells the worker to add GKD's supervised CE term on the dataset completion.
                signal = dict(cluster.worker.forward_kwargs(update, False))
                state.handler.on_step_begin()
                cluster.model.forward_backward(
                    inputs=copy_inputs(update), gradient_accumulation_steps=1,
                    micro_batch_size=micro_batch_size, **signal)
                state.handler.on_pre_optimizer_step()
                cluster.model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                                 gradient_accumulation_steps=1)
                state.handler.on_optimizer_step()
                state.record(extra_metrics=balance_metrics)
                state.step_tail(save=save)
        return False

    lent = False
    batches = 0
    state.handler.on_train_begin()
    try:
        if not train_off_policy():
            return
        # INITIAL PUBLICATION: the sampler must start from the student's weights.
        cluster.engine.sync_weights()
        lent = cluster.colocate
        for trajs, batch_facts in driver:
            if cluster.colocate:
                # COLOCATE HANDOVER: sleep the sampler, reload the trainer.
                sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                cluster.model.reload_to_gpu()
                lent = False
            lengths = [p + r for p, r in zip(batch_facts.get('prompt_lengths') or (),
                                             batch_facts.get('response_lengths') or ())]
            dropped = len(trajs) % update_batch_size
            if dropped:
                print(f'[gkd] dropping {dropped} rows that cannot fill an optimizer batch')
            # ON-POLICY UPDATE: the student's own rollouts, use_student=True (no extra CE term).
            for lo, hi in optimizer_batch_slices(len(trajs), train_batch_size, ga):
                if state.reached_max() or state.should_stop():
                    break
                update = trajs[lo:hi]
                state.micro_step += (hi - lo) // train_batch_size
                update, balance_metrics = balance_update(
                    state, update, lengths[lo:hi],
                    micro_batch_kwargs={'micro_batch_size': micro_batch_size})
                # ON-policy teacher signal: the worker scores the student's own response tokens
                # under the teacher view; the generalized JSD (beta interpolates the mixture).
                signal = dict(cluster.worker.forward_kwargs(update, True))
                state.handler.on_step_begin()
                cluster.model.forward_backward(
                    inputs=copy_inputs(update), gradient_accumulation_steps=1,
                    micro_batch_size=micro_batch_size, **signal)
                state.handler.on_pre_optimizer_step()
                cluster.model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                                 gradient_accumulation_steps=1)
                state.handler.on_optimizer_step()
                state.record(extra_metrics=balance_metrics)
                state.step_tail(save=save)
            batches += 1
            if not train_off_policy():
                break
            if batches % sync_every == 0:
                cluster.engine.sync_weights()
                lent = cluster.colocate
        state.handler.on_train_end()
        save_final_checkpoint(cluster.model, save, enabled=True)
    finally:
        close_driver_preserving_error(driver)
        if lent:
            sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
            cluster.model.reload_to_gpu()
        finalize_rl_resources(
            state, cluster, remote_run=False, driver=driver, recipe='GKD-example',
            finish=lambda: state.finish_metrics(status='loop_completed'))

    print(f'GKD finished: {state.global_step} optimizer steps')


if __name__ == '__main__':
    main()
