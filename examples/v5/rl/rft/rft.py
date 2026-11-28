# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 RFT at the COMPONENT level -- the code twin of rft.sh, with the whole training process exposed.
#
# RFT (Rejection-sampling Fine-Tuning): a multi-round loop. Each round (rft_iterations=3 here):
#   1. push the just-trained policy into the sampler (one weight sync);
#   2. generate rft_num_samples=8 completions per prompt over the WHOLE prompt set, scoring each
#      with the ORM ('accuracy' = MathAccuracy against the dataset's solution column);
#   3. SELECT the kept completions per prompt (rft_select='best_of_n' keeps the single highest-
#      reward sample; 'top_k'/'threshold' are the other rules -- RFTSelector below);
#   4. SFT the kept completions (plain cross-entropy -- no advantage, no log-prob ratios: the
#      rejection filter IS the learning signal).
# The policy improves round over round because each round distills the model's own best samples.
# Convergence signal: train/total_reward climbing at ROUND boundaries (the reward is one value per
# round -- RFT re-samples only between rounds); the SFT loss on the kept subset is NOT the signal.
#
# THE FIVE SEGMENTS mirror run_rft: prompts -> devices -> cluster (policy + sampler + ORM, no
# reference/critic) -> dataloader + selector + StepState -> the ROUND loop written out below.
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors rft.sh (colocate: the sampler time-shares the
# trainers' GPUs; each round's sync hands them over, the round's end hands them back).
# SIMPLIFIED vs the recipe: checkpoints here use the ordinary periodic save (the recipe defers
# saves to round boundaries and commits a round component so resume replays whole rounds --
# round-atomic resume is engineering scaffolding, see run_rft._train_selected_samples); no resume /
# perf profiling. Kept: the selection rule, the weighted-reward fold, the round sync cadence, the
# tail drop to whole optimizer batches, and the 0-step fail-loud guard.
#
# Run: CUDA_VISIBLE_DEVICES=<2 free cards> /usr/local/bin/python examples/v5/rl/rft/rft.py

import math

import torch

from swift.dev.builder import build_prompt_dataloader, resolve_rl_dp_size
from swift.dev.builder.optimizer import resolve_max_grad_norm
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    GenerationConfig,
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
from swift.dev.recipe.offload import sleep_for_rollout
from swift.dev.recipe.rft.assembly import build_rft_driver, initialize_rft_cluster, rft_group_requests
from swift.dev.recipe.rft.run_rft import RFTSelector
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
)
from swift.dev.utils.scoring import copy_inputs, weighted_rewards
# MEASURED (2026-11-24, 2-card Qwen3.5-0.8B smoke of the sibling rft.sh, same knobs): FAIL, not a
# clean pass -- ~3 GiB/card over the first 2 steps, then ReferenceCountingAssertionError.


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as rft.sh."""
    return {
        'model_config': ModelConfig(attn_impl='flash_attention_2', model='Qwen/Qwen3.5-4B', torch_dtype='bfloat16'),
        'template_config': TemplateConfig(
            max_length=3072,
            system='You are a helpful math assistant. Solve the problem step by step and put your '
                   'final answer within \\boxed{}.'),
        'dataset_config': DatasetConfig(dataset=['AI-MO/NuminaMath-TIR#2000']),
        'train_config': TrainConfig(
            num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            learning_rate=1e-4, warmup_ratio=0.05, gradient_checkpointing=True),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        'generation_config': GenerationConfig(temperature=1.0),
        # THE ALGORITHM KNOBS: rft_num_samples = completions generated per prompt per round;
        # rft_select = the rejection rule (best_of_n keeps the single best); rft_iterations = rounds.
        # No beta/epsilon/num_iterations: RFT has no KL term, no clipping, no batch replay.
        'rl_config': RLConfig(
            rl_type='rft', rft_num_samples=8, rft_select='best_of_n', rft_iterations=3,
            max_completion_length=2048),
        # THE ROLLOUT KNOBS: colocate shares the trainer GPUs with the sampler; each round syncs
        # once at its start (the whole prompt set is generated under ONE policy version per round).
        'rollout_config': RolloutConfig(sampler_mode='colocate', rollout_sampler='vllm',
                                        vllm_gpu_memory_utilization=0.5),
        # THE REWARD CHANNEL: 'accuracy' scores each completion against the solution column.
        'reward_config': RewardConfig(orm=['accuracy']),
        'quantize_config': QuantizeConfig(),
    }


def main() -> None:
    cfg = build_configs()
    model_config, template_config = cfg['model_config'], cfg['template_config']
    dataset_config, train_config = cfg['dataset_config'], cfg['train_config']
    distributed_config, checkpoint_config = cfg['distributed_config'], cfg['checkpoint_config']
    rollout_config, rl_config = cfg['rollout_config'], cfg['rl_config']
    reward_config, tuner_config = cfg['reward_config'], cfg['tuner_config']
    logging_config, generation_config = cfg['logging_config'], cfg['generation_config']
    quantize_config = cfg['quantize_config']
    output_dir = checkpoint_config.output_dir
    ga = train_config.gradient_accumulation_steps

    # SEGMENT 1 -- PROMPTS: the whole prompt set is re-generated EVERY round (shuffle=False below).
    prompt_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)

    # SEGMENT 2 -- DEVICES: twinkle initialize + template + the device-group plan; rft_group_requests
    # declares any frozen reward-MODEL groups (a rule ORM like 'accuracy' needs none).
    template, resource_plan, _ = initialize_rl_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=None, moe_config=None,
        output_dir=output_dir, prompt_rows=prompt_rows,
        group_requests=(lambda: rft_group_requests(reward_config)))

    # BATCH GEOMETRY + STEP BUDGET: num_samples per prompt; best_of_n keeps exactly 1, so the
    # budget's effective num_generations is 1 (top_k/threshold would keep more -- see
    # RFTSelector._select_keep). Each round SFTs the kept set once; rft_iterations rounds.
    num_samples = rl_config.rft_num_samples or rl_config.num_generations
    dp_size = resolve_rl_dp_size(distributed_config, template_config.sequence_parallel_size)
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    generation_batch_size = resolve_generation_batch_size(
        rollout_config.generation_batch_size, num_prompts=len(prompts),
        num_generations=num_samples, num_iterations=1,
        train_batch_size=train_batch_size, gradient_accumulation_steps=ga)
    kept_per_prompt = 1  # best_of_n; raise for top_k/threshold to size the budget correctly
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts), num_generations=kept_per_prompt,
            train_batch_size=train_batch_size, gradient_accumulation_steps=ga,
            num_train_epochs=rl_config.rft_iterations, num_iterations=1,
            generation_batch_size=generation_batch_size),
        recipe='rft-example')

    # SEGMENT 3 -- THE CLUSTER: policy + weight-syncable sampler + ORM scorers. NO reference (no KL
    # term) and NO critic (no value baseline) -- rejection sampling needs neither.
    cluster = initialize_rft_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        quantize_config=quantize_config, megatron_config=None, moe_config=None,
        generation_config=generation_config, resource_plan=resource_plan,
        inflight=generation_batch_size * num_samples)

    # SEGMENT 4 -- DATALOADER + SELECTOR + STEP CADENCE. Every round reads the whole prompt set in
    # order; the selector owns the rejection rule AND the round-level reward metric (its
    # step_telemetry feeds train/total_reward -- one value per round, by construction).
    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=dataset_config.data_seed, shuffle=False,
        batch_size=generation_batch_size)
    selector = RFTSelector(
        num_generations=num_samples, rft_select=rl_config.rft_select,
        rft_threshold=rl_config.rft_threshold, rft_top_k=rl_config.rft_top_k,
        rft_max_samples_per_prompt=rl_config.rft_max_samples_per_prompt,
        reward_scorer_names=cluster.orm_scorer_names, reward_weights=reward_config.orm_weights)
    state = StepState(
        cluster.model, train_config, padding_free=template_config.padding_free,
        dp_balance=train_config.dp_balance, gradient_accumulation_steps=ga, max_steps=max_steps,
        max_grad_norm=resolve_max_grad_norm(train_config), logging_config=logging_config,
        output_dir=output_dir, save_steps=checkpoint_config.save_steps,
        extra_metrics_fn=selector.step_telemetry)
    save_knobs = build_save_knobs(checkpoint_config, dataset_config, distributed_config, tuner_config,
                                  output_dir=output_dir)

    def save(name: str, is_final: bool = False) -> str:
        state.publish_state()
        return save_training_step(cluster.model, state.handler, name, save_knobs,
                                  consumed_train_samples=state.global_step, is_final=is_final)

    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    # SEGMENT 5 -- THE VISIBLE ROUND LOOP.
    micro_batch_size = max(1, train_batch_size // dp_size)
    update_batch_size = train_batch_size * ga
    colocate = effective_colocate(rollout_config, resource_plan)
    state.handler.on_train_begin()
    try:
        for round_index in range(rl_config.rft_iterations):
            if state.reached_max():
                break
            # ROUND STEP 1 -- PUBLISH: the round generates under the just-trained policy; under
            # colocate the sync also hands the GPUs to the sampler.
            cluster.engine.sync_weights()
            # ROUND STEP 2 -- GENERATE + SCORE: one driver over the whole prompt set; collect the
            # trajectory references + the row-aligned ORM score column + response lengths.
            driver = build_rft_driver(cluster=cluster, rollout_config=rollout_config,
                                      prompt_dataloader=dataloader, num_samples=num_samples,
                                      window=generation_batch_size)
            trajs, orm_rows, lengths = [], [], []
            try:
                for batch, batch_facts in driver:
                    trajs.extend(batch)
                    for score in batch_facts.get('scores') or ():
                        orm_rows.append(list(((score.get('rewards') or {}).get('orm') or {}).values()))
                    lengths.extend(batch_facts.get('response_lengths') or ())
            finally:
                close_driver_preserving_error(driver)
                if colocate:
                    # Hand the GPUs back to the trainer for the round's SFT pass.
                    sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                    cluster.model.reload_to_gpu()
            if not trajs:
                raise RuntimeError('the RFT round produced no scored group (every episode failed).')
            # ROUND STEP 3 -- SELECT: fold the named ORM columns into one weighted reward per row
            # (nan-safe), feed the WHOLE rollout into the reward metric (so train/total_reward
            # tracks sample quality, not just the kept subset), then apply the rejection rule.
            rewards_matrix = torch.tensor(
                [[float('nan') if v is None or v != v else float(v) for v in (row or ())] or [float('nan')]
                 for row in orm_rows], dtype=torch.float32)
            selector.accumulate_metric(rewards_matrix, lengths)
            rewards = weighted_rewards(rewards_matrix, reward_config.orm_weights).tolist()
            kept = selector.select(len(trajs), rewards)
            kept_trajs = [trajs[i] for i in kept]
            kept_lengths = [lengths[i] for i in kept]
            print(f'[rft] round {round_index}: kept {len(kept_trajs)}/{len(trajs)} completions')
            # ROUND STEP 4 -- SFT THE KEPT SET: plain cross-entropy updates (no advantage/logps
            # columns exist in RFT); drop the tail that cannot fill a whole optimizer batch.
            dropped = len(kept_trajs) % update_batch_size
            if dropped:
                print(f'[rft] dropping {dropped} kept rows that cannot fill an optimizer batch')
            for lo, hi in optimizer_batch_slices(len(kept_trajs), train_batch_size, ga):
                if state.reached_max() or state.should_stop():
                    break
                update = kept_trajs[lo:hi]
                state.micro_step += (hi - lo) // train_batch_size
                update, balance_metrics = balance_update(
                    state, update, kept_lengths[lo:hi],
                    micro_batch_kwargs={'micro_batch_size': micro_batch_size})
                state.handler.on_step_begin()
                cluster.model.forward_backward(
                    inputs=copy_inputs(update), gradient_accumulation_steps=1,
                    micro_batch_size=micro_batch_size)
                state.handler.on_pre_optimizer_step()
                cluster.model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                                 gradient_accumulation_steps=1)
                state.handler.on_optimizer_step()
                state.record(extra_metrics=balance_metrics)
                state.step_tail(save=save)
            if state.reached_max() or state.should_stop():
                break
        if state.global_step == 0:
            # Fail loudly rather than hand back an untrained checkpoint (same guard as run_rft).
            raise RuntimeError(
                f'RFT took 0 optimizer steps across {rl_config.rft_iterations} round(s); no round kept '
                f'enough completions to fill a train_batch_size={train_batch_size} mini-batch.')
        state.handler.on_train_end()
        save_final_checkpoint(cluster.model, save, enabled=True)
    finally:
        finalize_rl_resources(
            state, cluster, remote_run=False, recipe='RFT-example',
            finish=lambda: state.finish_metrics(status='loop_completed'))
    print(f'RFT finished: {state.global_step} optimizer steps over {rl_config.rft_iterations} rounds')


if __name__ == '__main__':
    main()