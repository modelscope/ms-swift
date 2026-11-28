# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 OPSD at the COMPONENT level -- the code twin of opsd.sh, with the whole training process exposed.
#
# OPSD (On-Policy Self-Distillation): the STUDENT generates its own answers (on-policy rollouts),
# then a TEACHER scores those same answer tokens under a PRIVILEGED view -- the question PLUS a
# reference solution the student never sees (the dataset's `teacher_prompt` column). The student
# matches the teacher's per-token log-probabilities (signal='logps'; opsd_reverse=True selects the
# reverse-KL direction). No full-vocab logits transfer, no reward model: the teacher's logps ARE
# the training signal, so the loss falling is the convergence signal.
#
# THE FIVE SEGMENTS mirror run_distill (shared with gkd/mopd; rl_type picks the signal):
#   1. prompts -- the plugin derives `teacher_prompt` from each row's problem + solution;
#   2. devices (initialize_rl_devices + distill_group_requests for the frozen teacher);
#   3. cluster (initialize_distill_cluster: student, sampler, teacher, DistillWorker, engine);
#   4. driver (build_distill_driver: streams the student's rollouts, no reward scoring);
#   5. the loop (teacher logps signal -> forward_backward/clip_grad_and_step -> weight sync).
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors opsd.sh (colocate; the 4B teacher holds its own
# frozen group). SIMPLIFIED vs the recipe: lmbda=1.0 is pinned (purely on-policy), so the recipe's
# off-policy coin draw is not written here (it lives in run_distill._train_loop and in ../gkd/gkd.py);
# also absent: resume / perf profiling. The privileged-view signal, tail drop and sync cadence stay.
#
# Run: CUDA_VISIBLE_DEVICES=<2 free cards> /usr/local/bin/python examples/v5/rl/opsd/opsd.py
# Convergence signal: the loss falling (it IS the teacher-student divergence; there is no reward).

import math
from pathlib import Path

from swift.dev.utils.plugin import PluginRegistry
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
    """The run as typed Config objects -- the same knob values as opsd.sh."""
    return {
        'model_config': ModelConfig(attn_impl='flash_attention_2', model='Qwen/Qwen3.5-4B', torch_dtype='bfloat16'),
        'template_config': TemplateConfig(max_length=8192),
        # opsd_numina is registered by opsd_plugin.py (loaded in main): it streams NuminaMath-TIR
        # and derives each row's privileged `teacher_prompt` from the problem + reference solution.
        'dataset_config': DatasetConfig(dataset=['opsd_numina#5000']),
        'train_config': TrainConfig(
            num_train_epochs=1, per_device_train_batch_size=4, gradient_accumulation_steps=1,
            learning_rate=2e-5, warmup_ratio=0.05, gradient_checkpointing=True),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2,
                                              save_only_model=True),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        # THE ALGORITHM KNOBS: teacher_model = the frozen scorer; lmbda=1 (purely on-policy);
        # opsd_reverse = reverse-KL direction; num_generations=1 (no group -- no group advantage);
        # sampling_temperature scales the ROLLOUT (distinct from distill_temperature).
        'rl_config': RLConfig(
            rl_type='opsd', teacher_model=['Qwen/Qwen3.5-4B'],
            lmbda=1.0, opsd_reverse=True, num_generations=1, sampling_temperature=1.0,
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
    # PLUGIN LOAD (the CLI's --external_plugins equivalent): import opsd_plugin.py so its
    # @register_dataset calls run BEFORE the dataset name is resolved; load_external also
    # publishes the source to every Ray worker so remote actors can re-import it.
    PluginRegistry.load_external([str(Path(__file__).resolve().parent / 'opsd_plugin.py')])
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
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='opsd-example')

    # SEGMENT 3 -- THE CLUSTER: every actor distillation trains with, built explicitly. model = the
    # student policy (Ray actors, LoRA); sampler = the weight-syncable vLLM engine generating the
    # student's rollouts; teachers = the frozen scorer(s); worker = the DistillWorker that runs the
    # teacher forward per update and packages its signal as light training kwargs (the driver never
    # sees the teacher's tensors); engine = publishes student weights to the sampler.
    cluster = initialize_distill_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        tuner_config=tuner_config, max_steps=max_steps, signal=DISTILL_SIGNALS['opsd'],
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

    # SEGMENT 5 -- THE VISIBLE LOOP (lmbda=1: every block is an on-policy student batch).
    micro_batch_size = max(1, train_batch_size // dp_size)
    update_batch_size = train_batch_size * ga
    sync_every = rollout_config.parameter_sync_step
    # INITIAL PUBLICATION: the sampler must start from the student's weights (a LoRA-init policy
    # differs from what the engine loaded); under colocate this also hands the GPUs over.
    cluster.engine.sync_weights()
    lent = cluster.colocate
    batches = 0
    state.handler.on_train_begin()
    try:
        for trajs, batch_facts in driver:
            if cluster.colocate:
                # COLOCATE HANDOVER: sleep the sampler (offload its weights/KV), reload the trainer.
                sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                cluster.model.reload_to_gpu()
                lent = False
            lengths = [p + r for p, r in zip(batch_facts.get('prompt_lengths') or (),
                                             batch_facts.get('response_lengths') or ())]
            # TAIL DROP: only whole optimizer batches train; a remainder cannot fill
            # train_batch_size x ga rows and is dropped (the recipe logs the same warning).
            dropped = len(trajs) % update_batch_size
            if dropped:
                print(f'[opsd] dropping {dropped} rows that cannot fill an optimizer batch')
            # TRAIN MINI-BATCHES: slice into full updates, balance each across DP ranks by token
            # load, then ask the worker for the teacher signal and take the optimizer step.
            for lo, hi in optimizer_batch_slices(len(trajs), train_batch_size, ga):
                if state.reached_max() or state.should_stop():
                    break
                update = trajs[lo:hi]
                state.micro_step += (hi - lo) // train_batch_size
                update, balance_metrics = balance_update(
                    state, update, lengths[lo:hi],
                    micro_batch_kwargs={'micro_batch_size': micro_batch_size})
                # THE TEACHER SIGNAL (the heart of distillation): the worker runs the frozen
                # teacher over the SAME response tokens under the privileged teacher_prompt view and returns the signal as
                # light kwargs (refs, not tensors). use_student=True marks an on-policy update.
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
            # WEIGHT PUBLICATION: every sync_every batches, push the updated student weights to the
            # sampler so the next rollouts come from the new policy (on-policy freshness).
            batches += 1
            if batches % sync_every == 0 and not (state.reached_max() or state.should_stop()):
                cluster.engine.sync_weights()
                lent = cluster.colocate
            if state.reached_max() or state.should_stop():
                break
        state.handler.on_train_end()
        save_final_checkpoint(cluster.model, save, enabled=True)
    finally:
        # TEARDOWN in fixed order: close the driver first, then release every actor. If the loop
        # ended with the GPUs lent to the sampler, hand them back so the final save finds the model.
        close_driver_preserving_error(driver)
        if lent:
            sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
            cluster.model.reload_to_gpu()
        finalize_rl_resources(
            state, cluster, remote_run=False, driver=driver, recipe='OPSD-example',
            finish=lambda: state.finish_metrics(status='loop_completed'))

    print(f'OPSD finished: {state.global_step} optimizer steps')


if __name__ == '__main__':
    main()
