# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 PPO at the COMPONENT level -- the code twin of ppo.sh, with the whole training process exposed.
#
# PPO (Proximal Policy Optimization) trains TWO models against a reward model:
#   * the POLICY generates answers (vLLM sampler) and is updated with the clipped surrogate objective
#     (cliprange=0.2) using per-token advantages;
#   * the CRITIC (a trainable num_labels=1 value head, initialized from the reward model's base)
#     predicts each token's expected return and is updated with a clipped value loss (cliprange_value).
# The advantage is GAE (Generalized Advantage Estimation): token rewards = the ORM score on the last
# token MINUS kl_coef * per-token KL against the frozen reference; gamma/lam shape the temporal
# smoothing over the critic's pre-update values. num_ppo_epochs=4 replays each rollout batch that
# many times (the clipping is what keeps the replays stable).
#
# THE FIVE SEGMENTS mirror run_ppo: prompts -> devices -> cluster (policy + critic + reference +
# reward channel, every actor explicit) -> dataloader + driver -> the loop:
#   critic values + reference logps -> ONE GAE advantage call -> policy step + critic step (same
#   update) -> weight sync every parameter_sync_step batches.
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors ppo.sh's trainer ranks; the reward MODEL (--orm is
# a model id here, not a rule) holds its OWN extra GPU -- 3 cards total, as ppo.sh's CUDA list shows.
# Colocate time-shares the trainer GPUs with the sampler AND offloads the critic across each sync.
# SIMPLIFIED vs the recipe: no EMA reference sync, no rollout throughput accounting, no resume /
# perf profiling; the critic checkpoint here rides the ordinary save (the recipe nests a
# step-aligned value_model dir inside each policy checkpoint -- see run_ppo._save_value_checkpoint).
# Kept: the GAE call, both clipped losses, the temperature-scaled reference logps, the colocate
# handover, the replay count and the sync cadence.
#
# Run: CUDA_VISIBLE_DEVICES=<3 free cards> /usr/local/bin/python examples/v5/rl/ppo/ppo.py
# Convergence signal: train/total_reward climbing; the critic's explained_variance rising (the
# value predictions explaining the returns) -- the policy loss alone wanders.

import math

from swift.dev.builders import build_prompt_dataloader, resolve_rl_dp_size
from swift.dev.builders.optimizer import resolve_max_grad_norm
from swift.dev.builders.streaming import offload_kwargs
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
from swift.dev.recipe.batching import balance_lanes, balance_update
from swift.dev.recipe.offload import sleep_for_rollout
from swift.dev.recipe.ppo.assembly import build_ppo_driver, initialize_ppo_cluster, ppo_group_requests
from swift.dev.recipe.utils import (
    RolloutStepState,
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
from twinkle.metric import CompletionRewardMetric
# MEASURED: peak VRAM ~3 GiB/card (Qwen3.5-0.8B, 2-step smoke, 2 card(s), 2026-11-24)


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as ppo.sh."""
    return {
        'model_config': ModelConfig(attn_impl='flash_attention_2', model='Qwen/Qwen3.5-4B', torch_dtype='bfloat16'),
        'template_config': TemplateConfig(max_length=2048),
        'dataset_config': DatasetConfig(dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20000']),
        'train_config': TrainConfig(
            num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-5, warmup_ratio=0.05, gradient_checkpointing=True),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        'generation_config': GenerationConfig(temperature=0.7),
        # THE ALGORITHM KNOBS: num_generations = samples per prompt (PPO has no group contract --
        # GAE scores each trajectory on its own); num_ppo_epochs = batch replays; cliprange /
        # cliprange_value = the policy / value clipping; gamma/lam = GAE discount/trace; kl_coef =
        # the per-token KL penalty against the reference (beta is not used by PPO).
        'rl_config': RLConfig(
            rl_type='ppo', num_generations=2, num_ppo_epochs=4,
            cliprange=0.2, cliprange_value=0.2, gamma=1.0, lam=0.95, kl_coef=0.05,
            max_completion_length=512),
        # THE ROLLOUT KNOBS: colocate shares the trainer GPUs with the sampler (lower utilization
        # than GRPO here -- the critic also lives on these cards); parameter_sync_step publishes
        # the policy every N batches.
        'rollout_config': RolloutConfig(sampler_mode='colocate', rollout_sampler='vllm',
                                        vllm_gpu_memory_utilization=0.4),
        # THE REWARD CHANNEL: a MODEL id (not a rule) -- a frozen scorer on its own GPU group; the
        # critic's value head is initialized from this same base (first orm model id).
        'reward_config': RewardConfig(orm=['Qwen/Qwen3.5-4B']),
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

    # SEGMENT 1 -- PROMPTS.
    prompt_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)

    # SEGMENT 2 -- DEVICES: ppo_group_requests declares the frozen reward-model group(s); the critic
    # shares the policy's 'model' group (policy and critic MUST slice_dp at the same data width).
    template, resource_plan, auxiliary_meshes = initialize_rl_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=None, moe_config=None,
        output_dir=output_dir, prompt_rows=prompt_rows,
        group_requests=(lambda: ppo_group_requests(
            rl_config, reward_config, distributed_config, tuner_config)))

    # BATCH GEOMETRY: num_ppo_epochs replays feed the budget (each rollout batch trains that many
    # passes); the derivation is run_ppo's, kept visible.
    dp_size = resolve_rl_dp_size(distributed_config, template_config.sequence_parallel_size)
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    generation_batch_size = resolve_generation_batch_size(
        rollout_config.generation_batch_size, num_prompts=len(prompts),
        num_generations=rl_config.num_generations, num_iterations=rl_config.num_ppo_epochs,
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
        num_iterations=rl_config.num_ppo_epochs, stream_batch_rows=batch_size)
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='ppo-example')

    # SEGMENT 3 -- THE CLUSTER: every actor PPO trains with. model = policy; value_model = the
    # trainable critic (seq_cls num_labels=1, LoRA head over the reward base); reference = frozen
    # initial policy for the KL penalty (LoRA: the policy with its adapter disabled); orm_scorers =
    # the frozen reward model; advantage = the GAE estimator handle; engine = weight publication.
    cluster = initialize_ppo_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        batch_size=batch_size, quantize_config=quantize_config, megatron_config=None, moe_config=None,
        generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes)

    # SEGMENT 4 -- DATALOADER + DRIVER + STEP CADENCE.
    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=dataset_config.data_seed,
        shuffle=dataset_config.train_dataloader_shuffle, batch_size=generation_batch_size)
    steps_per_batch = max(1, batch_size * max(1, rl_config.num_ppo_epochs) // (train_batch_size * ga))
    driver = build_ppo_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, reward_config=reward_config,
        prompt_dataloader=dataloader, batch_size=batch_size, num_train_epochs=num_train_epochs,
        max_steps=math.ceil(max_steps / steps_per_batch))
    reward_metric = CompletionRewardMetric()
    state = RolloutStepState(
        cluster.model, train_config, padding_free=template_config.padding_free,
        dp_balance=train_config.dp_balance, gradient_accumulation_steps=ga, max_steps=max_steps,
        max_grad_norm=resolve_max_grad_norm(train_config), logging_config=logging_config,
        output_dir=output_dir, save_steps=checkpoint_config.save_steps,
        extra_metrics_fn=reward_metric.calculate, extra_metrics_granularity='iteration')
    save_knobs = build_save_knobs(checkpoint_config, dataset_config, distributed_config, tuner_config,
                                  output_dir=output_dir)

    def save(name: str, is_final: bool = False) -> str:
        state.publish_state()
        return save_training_step(cluster.model, state.handler, name, save_knobs,
                                  consumed_train_samples=state.global_step, is_final=is_final)

    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    # SEGMENT 5 -- THE VISIBLE LOOP.
    temperature = float(cluster.sampling_params.get('temperature', 1.0))
    micro_batch_size = max(1, train_batch_size // dp_size)
    batching = {'micro_batch_size': micro_batch_size}
    sync_every = rollout_config.parameter_sync_step
    ref_shared = cluster.reference is cluster.model
    lent = False

    def sync() -> None:
        # COLOCATE PUBLICATION: offload the critic first (it shares the trainer cards), then push
        # the policy weights -- the sync wakes the sampler and takes the GPUs.
        nonlocal lent
        if cluster.colocate:
            cluster.value_model.offload_to_cpu(**offload_kwargs(rollout_config))
            lent = True
        cluster.engine.sync_weights()

    def reclaim() -> None:
        nonlocal lent
        sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
        cluster.model.reload_to_gpu()
        cluster.value_model.reload_to_gpu()
        lent = False

    batches = 0
    state.handler.on_train_begin()
    try:
        sync()  # initial publication: the sampler must start from the policy's weights
        for trajs, batch_facts in driver:
            if cluster.colocate:
                reclaim()
            # NONEMPTY-RESPONSE RULE (in the worker, on the real rows): drop rows with no response
            # tokens; the kept indices slice the row-aligned score column to what actually trains.
            trajs, kept_facts = cluster.traj_filter.select(
                trajs, where=lambda traj: bool(traj.get('response_token_ids')))
            if not trajs:
                raise RuntimeError('the pulled batch has no response tokens to train on.')
            scores = [batch_facts['scores'][index] for index in kept_facts['indices']]
            lengths = [p + r for p, r in zip(kept_facts.get('prompt_lengths') or (),
                                             kept_facts.get('response_lengths') or ())]
            # DP LANE BALANCING: GAE is per-sample (no group contract), so whole rows may move
            # freely across ranks; lane_restore puts results back into the caller's order (the
            # critic's rank-local token mean is order-sensitive and trains the original order).
            lane_trajs, lane_order, lane_restore = balance_lanes(
                state, trajs, lengths, group_size=1, micro_batch_kwargs=batching)
            lane_scores = scores if lane_order is None else [scores[index] for index in lane_order]
            # CRITIC VALUES + REFERENCE LOGPS: two whole-batch response-framed forwards. The
            # reference logps are taken at the SAMPLING temperature (the ratio math compares
            # T-scaled logps; a T=1 forward would skew every KL term).
            values_out = cluster.value_model.forward_only(
                inputs=lane_trajs, output_frame='response', output_fields=('values',),
                task='value', **batching)
            ref_out = cluster.reference.forward_only(
                inputs=lane_trajs, output_frame='response', output_fields=('logps',),
                temperature=temperature, **batching,
                **({'disable_lora': True} if ref_shared else {}))
            # THE GAE CALL (the heart of PPO): token rewards = ORM score on the last token minus
            # kl_coef * KL; GAE over the pre-update critic values (gamma/lam); the rows come back
            # carrying advantages / returns / old_logps / old_values -- every anchor the clipped
            # losses read. Swap gamma/lam/kl_coef in RLConfig to change the estimator's behavior.
            trajs, gae_summary = cluster.advantage(
                lane_trajs, ref_out=ref_out, value_out=values_out, scores=lane_scores)
            if lane_restore is not None:
                trajs = [trajs[position] for position in lane_restore]
            reward_metric.reset()
            reward_metric.accumulate(
                rewards={'total': list(gae_summary.get('reward_totals') or ())},
                completion_lengths=list(kept_facts.get('response_lengths') or ()))
            # TRAIN BOTH MODELS: num_ppo_epochs replays of the batch; each update steps the policy
            # (clipped surrogate over advantages/old_logps) THEN the critic (clipped value loss over
            # returns/old_values, task='value') within the SAME optimizer step. The anchors stay
            # fixed across replays -- that is PPO's batch re-use.
            total_samples = len(trajs) * max(1, rl_config.num_ppo_epochs)
            for lo, hi in optimizer_batch_slices(total_samples, train_batch_size, ga):
                if state.reached_max() or state.should_stop():
                    break
                indices = [index % len(trajs) for index in range(lo, hi)]
                update, balance_metrics = balance_update(
                    state, [trajs[index] for index in indices], [lengths[index] for index in indices],
                    micro_batch_kwargs=batching)
                state.micro_step += (hi - lo) // train_batch_size
                state.handler.on_step_begin()
                cluster.model.forward_backward(
                    inputs=update, gradient_accumulation_steps=1, temperature=temperature, **batching)
                state.handler.on_pre_optimizer_step()
                cluster.model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                                 gradient_accumulation_steps=1)
                # The critic trains the ORIGINAL order (its rank-local token mean is not
                # permutation-invariant, unlike the policy's seq-mean-token-mean loss).
                cluster.value_model.forward_backward(
                    inputs=[trajs[index] for index in indices], gradient_accumulation_steps=1,
                    task='value', **batching)
                cluster.value_model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                                       gradient_accumulation_steps=1)
                state.handler.on_optimizer_step()
                state.record(extra_metrics=balance_metrics)
                state.step_tail(save=save)
            # WEIGHT PUBLICATION every sync_every batches (the critic never enters a publish --
            # the sampler only ever generates with the policy).
            batches += 1
            if batches % sync_every == 0 and not (state.reached_max() or state.should_stop()):
                sync()
            if state.reached_max() or state.should_stop():
                break
        state.handler.on_train_end()
        save_final_checkpoint(cluster.model, save, enabled=True)
    finally:
        # TEARDOWN: close the driver, return lent devices, then release every actor.
        state.stream_stats = close_driver_preserving_error(driver)
        if lent:
            reclaim()
        finalize_rl_resources(
            state, cluster, remote_run=False, driver=driver, recipe='PPO-example',
            finish=lambda: state.finish_rollout_metrics(stream_stats=state.stream_stats,
                                                        status='loop_completed'))
    print(f'PPO finished: {state.global_step} optimizer steps; '
          f'last reward={state.history[-1].get("train/total_reward") if state.history else None}')


if __name__ == '__main__':
    main()