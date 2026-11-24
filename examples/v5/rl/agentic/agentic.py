# Copyright (c) ModelScope Contributors. All rights reserved.
# STATUS: REFERENCE-ONLY -- assembly seams signature-verified against the live code;
# not smoke-run. Config values mirror the sibling .sh.
# v5 AGENTIC GRPO at the COMPONENT level -- the code twin of agentic.sh, with the training process
# exposed. The ALGORITHM is GRPO (see ../grpo/grpo.py for the five-segment commentary); what this
# example adds is the MULTI-TURN TOOL LOOP inside each rollout:
#   * tools=['sandbox'] binds the built-in sandbox tool plugin; the model may call run_command /
#     write_file / read_file inside a per-episode LocalEnv workspace (sandbox_kind='local' default);
#   * max_turns=3 caps the tool-call rounds per episode; max_trajectory_tokens=3072 caps the whole
#     multi-turn trajectory; sandbox_num_envs=2 leases two concurrent environments;
#   * the ORM ('accuracy') still scores the FINAL answer -- the sandbox executes code but gives no
#     reward itself; the group-relative advantage is computed over the finished trajectories.
# The driver streams multi-turn episodes exactly like single-turn ones (the turn loop lives inside
# the rollout), so the five segments and the training loop below are IDENTICAL to ../grpo/grpo.py.
# log_completions=True (a GRPO-only control) logs the sampled trajectories for inspection.
#
# RAY MODE: mode='ray', nproc_per_node=2 mirrors agentic.sh (colocate). The sandbox envs are CPU
# workers (LocalEnv runs commands on the host); switch sandbox_kind to 'openenv' + sandbox_env_name
# to plug a custom environment package (see agentic_openenv.sh and coding_env/ next to this file).
# SIMPLIFIED vs the recipe: same set as ../grpo/grpo.py (no teacher/old-logps lanes, routing replay,
# CHORD, EMA, dynamic-sample regeneration, resume, perf metrics).
#
# Run: CUDA_VISIBLE_DEVICES=<2 free cards> /usr/local/bin/python examples/v5/rl/agentic/agentic.py
# Convergence signal: train/total_reward climbing (the loss wanders -- reward is the RL signal).

import math

from swift.dev.builders import build_dp_mesh, build_prompt_dataloader
from swift.dev.builders.optimizer import resolve_max_grad_norm
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
from swift.dev.recipe.grpo.assembly import build_driver, grpo_group_requests, initialize_cluster
from swift.dev.recipe.offload import sleep_for_rollout
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
from swift.dev.utils.scoring import copy_inputs
from twinkle.metric import CompletionRewardMetric
# MEASURED: not run — HUNG: no steps after 2min (2026-11-24)


def build_configs() -> dict:
    """The run as typed Config objects -- the same knob values as agentic.sh."""
    return {
        'model_config': ModelConfig(attn_impl='flash_attention_2', model='Qwen/Qwen3.5-4B', torch_dtype='bfloat16'),
        'template_config': TemplateConfig(
            max_length=4096,
            system='You are a helpful math assistant. You may run Python in a sandbox to verify '
                   'computations. Put your final answer within \\boxed{}.'),
        'dataset_config': DatasetConfig(dataset=['AI-MO/NuminaMath-TIR#5000']),
        'train_config': TrainConfig(
            num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=4,
            learning_rate=1e-4, warmup_ratio=0.05, gradient_checkpointing=True),
        'distributed_config': DistributedConfig(mode='ray', nproc_per_node=2),
        'checkpoint_config': CheckpointConfig(output_dir='output', save_steps=100, save_total_limit=2),
        'logging_config': LoggingConfig(report_to=['tensorboard'], logging_steps=1),
        'tuner_config': TunerConfig(tuner='lora', lora_rank=8, lora_alpha=32, target_modules=['all-linear']),
        'generation_config': GenerationConfig(temperature=1.0),
        # THE ALGORITHM KNOBS: num_generations = group size (advantage normalization happens within
        # each group); epsilon = PPO clip range; beta = KL strength against the reference (beta>0 is
        # what builds cluster.reference); num_iterations = policy replays per rollout batch.
        'rl_config': RLConfig(
            rl_type='grpo', num_generations=8, advantage_estimator='grpo',
            epsilon=0.2, num_iterations=1, beta=0.04, max_completion_length=1024,
            log_completions=True),
        # THE ROLLOUT KNOBS: colocate placement as in grpo.py, PLUS the multi-turn sandbox face:
        # tools=['sandbox'] binds the env's tools; max_turns caps tool rounds per episode;
        # max_trajectory_tokens caps the whole trajectory; sandbox_num_envs leases concurrent envs;
        # command_timeout/memory_limit_gb bound each sandboxed command (LocalEnv knobs).
        'rollout_config': RolloutConfig(
            sampler_mode='colocate', rollout_sampler='vllm', vllm_gpu_memory_utilization=0.5,
            tools=['sandbox'], max_turns=3, max_trajectory_tokens=3072,
            sandbox_num_envs=2, sandbox_command_timeout=60, sandbox_memory_limit_gb=2.0),
        # THE REWARD CHANNELS: orm names the rule scorers ('accuracy' = MathAccuracy over the
        # dataset's solution column); a model id here would build a frozen reward model instead.
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

    # SEGMENT 1 -- PROMPTS: read the raw rows; the resource estimator below sizes device groups off
    # the prompt lengths, and the prompt dataloader streams these messages to the rollout driver.
    prompt_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)

    # SEGMENT 2 -- DEVICES: twinkle initialize (Ray), the template, and the device-group plan for
    # every actor. group_requests declares what the cluster will need (policy/sampler/reference/
    # scorers) so the planner can reserve disjoint or colocated GPUs before anything is built.
    template, resource_plan, auxiliary_meshes = initialize_rl_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=None, moe_config=None,
        output_dir=output_dir, prompt_rows=prompt_rows,
        group_requests=(lambda: grpo_group_requests(rl_config, reward_config, distributed_config, tuner_config)))

    # BATCH GEOMETRY: one optimizer update trains train_batch_size = per_device x dp rows; a rollout
    # batch must fill whole updates (generation_batch_size), and the step budget derives from the
    # prompt count x group size -- the same derivation run_grpo uses, kept visible.
    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    generation_batch_size = resolve_generation_batch_size(
        rollout_config.generation_batch_size, num_prompts=len(prompts),
        num_generations=rl_config.num_generations, num_iterations=rl_config.num_iterations,
        train_batch_size=train_batch_size, gradient_accumulation_steps=ga)
    colocate = effective_colocate(rollout_config, resource_plan)
    batch_size = stream_batch_size(
        colocate=colocate, generation_batch_size=generation_batch_size,
        num_generations=rl_config.num_generations, train_batch_size=train_batch_size,
        gradient_accumulation_steps=ga)
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size, gradient_accumulation_steps=ga,
        num_train_epochs=math.ceil(train_config.num_train_epochs),
        generation_batch_size=generation_batch_size, num_iterations=rl_config.num_iterations,
        stream_batch_rows=batch_size)
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='grpo-example')

    # SEGMENT 3 -- THE CLUSTER: every actor GRPO trains with, built explicitly. model = the policy
    # (Ray actors); sampler = the weight-syncable vLLM engine; reference = the frozen initial policy
    # for the KL term (with LoRA: the policy with its adapter disabled); orm_scorers = the reward
    # functions ('accuracy'); advantage = the estimator handle (group-relative math + KL-in-reward);
    # engine = the checkpoint engine that publishes policy weights to the sampler.
    cluster = initialize_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        batch_size=batch_size, quantize_config=quantize_config, megatron_config=None, moe_config=None,
        generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes, teacher_prompt=False)

    # SEGMENT 4 -- DATALOADER + DRIVER: the prompt stream feeds twinkle's StreamingDriver, which
    # generates num_generations completions per prompt on the sampler, scores them with the ORM
    # scorers, and hands out (trajectories, batch_facts) once whole groups are ready. batch_facts
    # carries the light metadata (lengths, scores) -- the trajectories themselves stay as references.
    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=dataset_config.data_seed,
        shuffle=dataset_config.train_dataloader_shuffle, batch_size=generation_batch_size)
    steps_per_batch = max(1, batch_size * max(1, rl_config.num_iterations) // (train_batch_size * ga))
    driver = build_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, reward_config=reward_config,
        prompt_dataloader=dataloader, batch_size=batch_size,
        num_train_epochs=math.ceil(train_config.num_train_epochs),
        max_steps=math.ceil(max_steps / steps_per_batch))

    # STEP CADENCE + CHECKPOINT: same components as the SFT example; the reward aggregate below is
    # the RL-specific extra metric (train/total_reward is the convergence signal, not the loss).
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
                                  consumed_train_samples=state.global_step,
                                  component_states={'training': {'micro_step': state.micro_step}},
                                  is_final=is_final)

    def record_batch_rewards(facts: dict) -> None:
        # Feed this batch's named ORM scores into the reward aggregate (equal-weight simplification
        # of run_grpo._record_reward_metric, which also handles orm_weights and nan verdicts).
        rows = [((row.get('rewards') or {}).get('orm') or {}) for row in (facts.get('scores') or ())]
        if not rows or not rows[0]:
            return
        names = list(rows[0])
        rewards = {name: [float(row[name]) for row in rows if row.get(name) is not None] for name in names}
        rewards['total'] = [sum(float(row[name]) for name in names if row.get(name) is not None) for row in rows]
        reward_metric.reset()
        reward_metric.accumulate(rewards=rewards, completion_lengths=list(facts.get('response_lengths') or ()))

    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    # SEGMENT 5 -- THE VISIBLE LOOP.
    temperature = float(cluster.sampling_params.get('temperature', 1.0))
    micro_batch_size = max(1, train_batch_size // dp_size)
    batching = {'micro_batch_size': micro_batch_size}
    sync_every = rollout_config.parameter_sync_step
    # INITIAL PUBLICATION: the sampler must start from the policy's weights (a resumed/LoRA-init
    # policy differs from what the engine loaded); under colocate this also hands the GPUs over.
    cluster.engine.sync_weights()
    batches = 0
    state.handler.on_train_begin()
    try:
        for trajs, batch_facts in driver:
            if cluster.colocate:
                # COLOCATE HANDOVER: sleep the sampler (offload its weights/KV), reload the trainer.
                sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                cluster.model.reload_to_gpu()
            record_batch_rewards(batch_facts)

            # DP LANE BALANCING: split the batch into per-rank lanes of equal token load, keeping
            # whole GRPO groups together (the advantage's positional group math needs complete
            # blocks); lane_restore puts results back into the caller's order afterwards.
            lane_lengths = [p + r for p, r in zip(batch_facts.get('prompt_lengths') or (),
                                                 batch_facts.get('response_lengths') or ())]
            lane_trajs, lane_order, lane_restore = balance_lanes(
                state, trajs, lane_lengths, group_size=rl_config.num_generations,
                micro_batch_kwargs=batching, episode_ids=batch_facts.get('episode_ids'))
            lane_scores = batch_facts['scores'] if lane_order is None else [
                batch_facts['scores'][index] for index in lane_order]
            # DP PADDING: forward lanes need equal per-rank row counts; pad with inert dummy rows
            # (advantage 0, stripped before training) up to a multiple of the data-parallel size.
            pad_count = (-len(lane_trajs)) % dp_size
            if pad_count:
                dummies = cluster.traj_update.pad_rows(lane_trajs[0], pad_count=pad_count)
                lane_trajs = list(lane_trajs) + dummies
                lane_scores = list(lane_scores) + [{} for _ in dummies]

            # REFERENCE LOGPS: one whole-batch response-framed forward on the frozen reference.
            # temperature divides the logits so these logps match the sampler's T-scaled ones --
            # skipping it skews every importance ratio/KL term. LoRA reference IS the policy object:
            # disable_lora=True reads the base weights instead of the adapter's.
            ref_out = None
            if cluster.reference is not None:
                ref_out = cluster.reference.forward_only(
                    inputs=lane_trajs, output_frame='response', output_fields=('logps',),
                    temperature=temperature, **batching,
                    **({'disable_lora': True} if cluster.reference is cluster.model else {}))

            # THE ADVANTAGE CALL: group-relative normalization of the ORM scores (+ KL-in-reward
            # against ref_logps when beta>0), attaching the training-target columns to the rows.
            # Swap advantage_estimator in RLConfig to change the math (reinforce++/rloo/...).
            trajs, _adv_summary = cluster.advantage(
                lane_trajs, ref_out=ref_out, global_step=state.global_step, scores=lane_scores)
            if pad_count:
                trajs = trajs[:len(trajs) - pad_count]
            if lane_restore is not None:
                trajs = [trajs[position] for position in lane_restore]

            # TRAIN MINI-BATCHES: slice the batch into full optimizer updates (train_batch_size x ga
            # rows each; num_iterations replays the batch), balance each update across ranks, then
            # forward_backward + clip_grad_and_step. Every training signal (advantages/old_logps/
            # ref_logps) rides the trajectory rows themselves -- the kwargs only carry batching.
            total_samples = len(trajs) * max(1, rl_config.num_iterations)
            for lo, hi in optimizer_batch_slices(total_samples, train_batch_size, ga):
                if state.reached_max() or state.should_stop():
                    break
                indices = [index % len(trajs) for index in range(lo, hi)]
                update, balance_metrics = balance_update(
                    state, [trajs[index] for index in indices], [lane_lengths[index] for index in indices],
                    micro_batch_kwargs=batching)
                state.handler.on_step_begin()
                model_kwargs = {'inputs': copy_inputs(update), 'gradient_accumulation_steps': 1,
                                'micro_batch_size': micro_batch_size, 'temperature': temperature}
                cluster.model.forward_backward(**model_kwargs)
                state.handler.on_pre_optimizer_step()
                cluster.model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                                 gradient_accumulation_steps=1)
                state.handler.on_optimizer_step()
                state.micro_step += (hi - lo) // train_batch_size
                state.record(extra_metrics=balance_metrics)
                state.step_tail(save=save)

            # WEIGHT PUBLICATION: every sync_every batches, push the updated policy weights to the
            # sampler so the next rollouts are generated by the new policy (on-policy freshness).
            batches += 1
            if batches % sync_every == 0 and not (state.reached_max() or state.should_stop()):
                cluster.engine.sync_weights()
            if state.reached_max() or state.should_stop():
                break
        state.handler.on_train_end()
        save_final_checkpoint(cluster.model, save, enabled=True)
    finally:
        # TEARDOWN in fixed order: close the driver (its terminal stats feed the summary), then
        # release every actor; a cleanup error must not mask an in-flight training exception.
        state.stream_stats = close_driver_preserving_error(driver)
        finalize_rl_resources(
            state, cluster, remote_run=False, driver=driver, recipe='GRPO-example',
            finish=lambda: state.finish_rollout_metrics(stream_stats=state.stream_stats,
                                                        status='loop_completed'))
    print(f'Agentic GRPO finished: {state.global_step} optimizer steps; '
          f'last reward={state.history[-1].get("train/total_reward") if state.history else None}')


if __name__ == '__main__':
    main()
