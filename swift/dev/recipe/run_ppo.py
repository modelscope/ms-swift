"""PPO assembly: run_ppo orchestration (proximal policy optimization).

Peer of ``run_grpo`` for the one online RL type with a learned value function. PPO here is built from
the pieces twinkle already provides, with no fabricated infrastructure:

  - policy loss = the shared clipped surrogate (``GRPOLoss`` with epsilon=cliprange) -- PPO's policy
    objective is exactly that clip, so ``configure_rlhf_loss`` maps ppo -> GRPOLoss;
  - critic = a ``seq_cls`` ``num_labels=1`` model forwarded with ``task='value'`` (which keeps the
    head's per-token output instead of pooling to the last token), emitting ``V(s_t)`` at every token,
    trained by the clipped value loss (``configure_ppo_value_loss`` -> twinkle ``PPOValueLoss``). The
    ``task='value'`` behaviour is symmetric across backends (transformers ``TransformersValuePatch`` /
    Megatron ``forward_step``);
  - reward = one or more frozen seq_cls reward models scored once per completion, plus the standard
    per-token KL-to-reference penalty;
  - rollout = the same weight-syncable ``SyncableRollout`` (colocate / disaggregated) run_grpo uses.

PER-TOKEN GAE: because the critic emits a value at every response token, PPO runs in its full
token-level form. Each response token gets a reward ``r_t = -kl_coef * (logp_t - ref_logp_t)`` with the
reward model's scalar added at the final token; twinkle's ``GAEAdvantage`` then walks backwards over
the response with ``gamma``/``lam`` to produce per-token advantages (for the clipped policy surrogate)
and returns (for the clipped value loss). advantages/old_logps/returns/old_values are all captured ONCE
per rollout and re-used across ``num_ppo_epochs``, so the epochs are genuine PPO batch re-uses.

The per-token value head is the DDP / AccelerateStrategy path; it reads the base model's last hidden
state and does not yet cover sequence-parallel / packed layouts, so the critic runs without those.

Placement (colocate vs heterogeneous) and weight-sync are identical to run_grpo.
"""
from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = logging.getLogger(__name__)

_IGNORE_INDEX = -100


def run_ppo(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rollout_config: RolloutConfig,
    rlhf_config: RLHFConfig,
    tuner_config: Optional[TunerConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    *,
    engine_args: Optional[Dict[str, Any]] = None,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run online PPO (policy + critic + reference + reward model + weight-syncable rollout).

    Reuses run_grpo's device planning and rollout, then trains the policy (clipped surrogate) and the
    critic (clipped value loss) over each rollout for ``num_ppo_epochs``. Returns the loss history.
    """
    from swift.dev.builders import build_sampler
    from swift.dev.loss import configure_ppo_value_loss, configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly
    from swift.dev.recipe.run_grpo import (
        SyncableRollout,
        _REF_GROUP,
        _grpo_sampling_params,
        _initialize_twinkle_rl,
        _prompts_from_dataset,
        _reward_group,
        _sampler_backend,
        _sampler_engine_args,
        _sampler_world_size,
        plan_rl_device_groups,
    )

    if rlhf_config.rlhf_type != 'ppo':
        raise ValueError(f'run_ppo requires rlhf_type="ppo", got {rlhf_config.rlhf_type!r}.')
    assembly = TrainAssembly(
        'run_ppo',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        rlhf_config=rlhf_config,
        output_dir=output_dir,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config)
    assembly.prepare()

    backend = _sampler_backend(rollout_config)
    sampler_world_size = _sampler_world_size(rollout_config, backend)
    # Plan DeviceGroups for PPO's FULL-PARAMETER frozen auxiliaries (RL_PLAN §3.5): the KL reference (a
    # frozen policy-init copy under full fine-tuning) and each reward model. A LoRA reference is the
    # adapter-disabled policy base ('disable_lora'), which reuses the policy actor and plans no group. The
    # names/world sizes here match what _build_reference / _build_reward_models build below.
    ref_disable_lora = tuner_config is not None and not rlhf_config.ref_adapters
    ref_remote_group: Optional[str] = None if ref_disable_lora else _REF_GROUP
    auxiliary_groups: List[Tuple[str, int]] = [] if ref_disable_lora else [(_REF_GROUP, 1)]
    auxiliary_groups += [(_reward_group(i), 1) for i in range(len(rlhf_config.reward_model or []))]
    groups, sampler_remote_group, colocate = plan_rl_device_groups(
        distributed_config.nproc_per_node,
        rollout_config.vllm_mode,
        sampler_world_size,
        auxiliary_groups=auxiliary_groups)
    # PPO runs SP=1: the per-token value-head critic does not cover sequence-parallel / packed layouts
    # (see the module docstring), and policy+critic MUST share one mesh -- a SP policy with an SP=1 critic
    # would slice_dp each forward_backward at a different data_world_size, starving the critic's ranks. So
    # PPO deliberately plans no SP mesh, and validate_configs rejects sequence_parallel_size>1 for it.
    _initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism)

    # The critic, the reference and the reward models all encode with the SAME template as the policy,
    # so a batch lines up token-for-token across the four forwards.
    template = assembly.build_template()
    # Prompts are rolled out, not iterated by a dataloader, so the step budget is derived from the prompt
    # set (B1). PPO counts one rollout as one step (_record_step per rollout), so its budget is the
    # generation-batch count for num_train_epochs passes; --max_steps still overrides.
    prompts = _prompts_from_dataset(dataset_config)
    from swift.dev.recipe.train_loop import prompt_batch_count, resolve_rollout_max_steps
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        prompt_batch_count(len(prompts), train_config.num_train_epochs, rollout_config.generation_batch_size),
        recipe='run_ppo')
    assembly.resolve_step_intervals(max_steps)

    # Design-B mini-batch width: per_device_train_batch_size * dp_size, the same global-batch formula GRPO
    # uses. Online RL is Ray-only and pure data-parallel over nproc_per_node, so dp_size comes off
    # build_ray_dp_mesh -- the exact mesh the policy/critic were placed on and slice_dp splits each
    # forward_backward across. Every mini-batch must carry >= dp_size rows or slice_dp leaves some rank
    # with no data ("Batch too small"), which is why PPO feeds train_batch_size-row mini-batches, not the
    # one-sample-per-step it used to.
    from swift.dev.builders import build_ray_dp_mesh
    dp_size = build_ray_dp_mesh(distributed_config).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size

    # Policy (trainer): the clipped surrogate is the PPO policy loss (configure_rlhf_loss maps ppo). PPO
    # plans no SP mesh (SP=1) -- the critic runs SP=1 too, so policy/critic/dp_size all agree on a pure-DP
    # mesh of data_world_size = nproc_per_node.
    assembly.build_model()
    model = assembly.model
    configure_rlhf_loss(model, rlhf_config)
    configure_optimizer(
        model, train_config, num_training_steps=max_steps, distributed_config=distributed_config)

    # Critic: a trainable seq_cls (num_labels=1) value model, trained by the clipped value loss.
    value_model = _build_value_model(
        model_config,
        rlhf_config,
        distributed_config,
        train_config,
        template,
        model_path=(os.path.join(assembly.resume_dir, 'value_model') if assembly.resume_dir else None),
        megatron_config=megatron_config,
        moe_config=moe_config)
    configure_ppo_value_loss(value_model, rlhf_config)
    configure_optimizer(
        value_model, train_config, num_training_steps=max_steps, distributed_config=distributed_config)

    # Rollout (weight-syncable, exactly as run_grpo), reward model(s) and reference for the KL penalty.
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=_sampler_engine_args(rollout_config, engine_args, colocate, backend),
        template=template,
        remote_group=sampler_remote_group)
    rollout = SyncableRollout(model, sampler, template, colocate=colocate)
    reference = _build_reference(
        model_config, tuner_config, template, rlhf_config, distributed_config, remote_group=ref_remote_group)
    reward_models = _build_reward_models(rlhf_config, template, distributed_config)

    loop = PPOLoop(
        model,
        value_model,
        rollout,
        reference,
        reward_models,
        prompts,
        rlhf_config=rlhf_config,
        max_steps=max_steps,
        generation_batch_size=rollout_config.generation_batch_size,
        train_batch_size=train_batch_size,
        num_train_epochs=train_config.num_train_epochs,
        seed=train_config.seed,
        gradient_accumulation_steps=assembly.ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        sampling_params=_grpo_sampling_params(rlhf_config, generation_config),
        logging_config=logging_config,
        output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        no_save_optim=checkpoint_config.no_save_optim or checkpoint_config.save_only_model,
        no_save_rng=checkpoint_config.no_save_rng or checkpoint_config.save_only_model,
        safe_serialization=checkpoint_config.safe_serialization,
        max_shard_size=checkpoint_config.max_shard_size,
        save_total_limit=checkpoint_config.save_total_limit,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0)
    assembly.loop = loop
    if assembly.resume_dir:
        policy_state = assembly.resume_model()
        value_state = assembly.resume_model(
            value_model, os.path.join(assembly.resume_dir, 'value_model'), adapter_name='')
        loop.resume(policy_state, value_state=value_state)
    try:
        history = loop.fit()
        if _save_final:
            assembly.save_final()
        return history
    finally:
        rollout.shutdown()


def _build_value_model(model_config: ModelConfig, rlhf_config: RLHFConfig, distributed_config: DistributedConfig,
                       train_config: TrainConfig, template: Any, *, model_path: Optional[str] = None,
                       megatron_config: Optional[MegatronConfig] = None,
                       moe_config: Optional[MoEConfig] = None) -> Any:
    """Build the trainable critic: a ``seq_cls`` num_labels=1 model, forwarded with ``task='value'``.

    Initialised from the first reward model when one is given (closest to TRL, which inits the value
    function from the reward model), else from the policy's own base. The value head IS the seq_cls
    ``score`` linear (trainable, in the optimizer) -- ``task='value'`` keeps its PER-TOKEN output
    ``V(s_t)`` instead of pooling to the last token, symmetric across transformers and Megatron. Placed
    with the SAME DistributedConfig as the policy.
    """
    from copy import copy

    from swift.dev.builders import build_model
    from swift.dev.processor import InputProcessor

    init_from = rlhf_config.reward_model[0] if rlhf_config.reward_model else model_config.model
    value_cfg = copy(model_config)
    value_cfg.model = model_path or init_from
    value_cfg.task_type = 'seq_cls'  # the per-token value rides the seq_cls score head
    value_cfg.num_labels = 1
    value_model = build_model(
        value_cfg,
        distributed_config,
        train_config,
        megatron_config=megatron_config,
        moe_config=moe_config)
    value_model.set_processor(InputProcessor, cp_partition_mode=distributed_config.cp_partition_mode)
    value_model.set_template(template)
    return value_model


def _build_reference(model_config: ModelConfig, tuner_config: Optional[TunerConfig], template: Any,
                     rlhf_config: RLHFConfig, distributed_config: DistributedConfig, *,
                     remote_group: Optional[str]) -> Any:
    """PPO's frozen reference for the KL penalty: 'disable_lora' (LoRA) or a frozen policy-init model.

    LoRA reuses the adapter-disabled base (no second model, no DeviceGroup); full fine-tuning loads a
    frozen copy of the policy's initial weights (PPO anchors the KL to the starting policy) as a separate
    Ray actor on its OWN ``remote_group`` DeviceGroup, built with the run's backend via
    :func:`frozen_auxiliary_distributed_config` (RL_PLAN §3.5, basic principles 1/2). ``remote_group`` is
    the 'ref' group the run body planned (None on the LoRA path, which returns before using it).
    """
    if tuner_config is not None and not rlhf_config.ref_adapters:
        return 'disable_lora'
    from copy import copy

    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.recipe.assembly import configure_frozen_adapter

    ref_config = copy(model_config)
    ref_config.model = rlhf_config.ref_model or model_config.model
    ref_config.model_type = rlhf_config.ref_model_type or model_config.model_type
    ref_config.model_revision = rlhf_config.ref_model_revision or model_config.model_revision
    ref = build_model(
        ref_config, frozen_auxiliary_distributed_config(distributed_config, 1), remote_group=remote_group)
    return configure_frozen_adapter(ref, template, rlhf_config.ref_adapters, role='ref')


def _build_reward_models(rlhf_config: RLHFConfig, template: Any,
                         distributed_config: DistributedConfig) -> List[Any]:
    """Build the frozen reward model(s) that score each completion (seq_cls, num_labels=1 scalar).

    Returns an empty list when none are configured -- the loop then relies solely on the KL/base reward
    and flags it. Each reward model is a FULL-PARAMETER frozen seq_cls scorer, so it is a separate Ray
    actor on its own ``reward_<i>`` DeviceGroup, built with the run's backend (RL_PLAN §3.5); the group
    names/world sizes (1 each) match what the run body planned through ``plan_rl_device_groups``.
    """
    if not rlhf_config.reward_model:
        return []
    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.config import ModelConfig
    from swift.dev.recipe.assembly import configure_frozen_adapter
    from swift.dev.recipe.run_grpo import _reward_group

    models: List[Any] = []
    for idx, rm_id in enumerate(rlhf_config.reward_model):
        rm_cfg = ModelConfig(model=rm_id, task_type='seq_cls')
        rm_cfg.num_labels = 1
        if rlhf_config.reward_model_type:
            rm_cfg.model_type = rlhf_config.reward_model_type[idx] if idx < len(rlhf_config.reward_model_type) else None
        if rlhf_config.reward_model_revision:
            rm_cfg.model_revision = (rlhf_config.reward_model_revision[idx]
                                     if idx < len(rlhf_config.reward_model_revision) else None)
        rm = build_model(
            rm_cfg, frozen_auxiliary_distributed_config(distributed_config, 1), remote_group=_reward_group(idx))
        adapters = [rlhf_config.reward_adapters[idx]] if rlhf_config.reward_adapters else []
        models.append(configure_frozen_adapter(rm, template, adapters, role='reward'))
    return models


class PPOLoop:
    """Per-token PPO loop: rollout -> per-token reward (KL + RM) -> GAE -> clipped policy + value.

    Each step rolls out completions from the weight-synced sampler, then for every response token builds
    a reward ``r_t = -kl_coef * (logp_t - ref_logp_t)`` (the reward model's scalar added at the final
    token) and runs twinkle's ``GAEAdvantage`` (constructed with ``gamma``/``lam``) over the response to
    get per-token advantages and returns. The policy is updated by the clipped surrogate (``GRPOLoss``
    over per-token ``advantages``/``old_logps``) and the critic by the clipped value loss
    (``PPOValueLoss`` over per-token ``returns``/``old_values``).
    All of advantages/old_logps/returns/old_values are captured ONCE per rollout, so the
    ``num_ppo_epochs`` inner passes are genuine PPO re-uses of the same batch.
    """

    def __init__(
        self,
        model: TrainableModel,
        value_model: TrainableModel,
        rollout: Any,
        reference: Any,
        reward_models: List[Any],
        prompts: List[List[dict]],
        *,
        rlhf_config: RLHFConfig,
        max_steps: int = 1,
        generation_batch_size: Optional[int] = None,
        train_batch_size: int = 1,
        num_train_epochs: float = 1.0,
        seed: int = 42,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        sampling_params: Optional[dict] = None,
        logging_config: Optional['LoggingConfig'] = None,
        output_dir: str = 'output',
        save_steps: Optional[int] = None,
        no_save_optim: bool = False,
        no_save_rng: bool = False,
        safe_serialization: bool = True,
        max_shard_size: str = '5GB',
        save_total_limit: Optional[int] = None,
        manual_gc: bool = False,
        manual_gc_steps: int = 0,
    ):
        self.model = model
        self.value_model = value_model
        self.rollout = rollout
        self.reference = reference
        self.reward_models = reward_models
        self.prompts = prompts
        self.rlhf_config = rlhf_config
        self.max_steps = max_steps
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        #: Design-B mini-batch width (per_device_train_batch_size * dp_size). Each ``forward_backward``
        #: carries exactly this many rollout rows so slice_dp hands every DP rank per_device_train_batch_size
        #: of them; see ``_plan_mini_batches``.
        self.train_batch_size = max(1, train_batch_size)
        self.max_grad_norm = max_grad_norm
        self.sampling_params = sampling_params
        self.logging_config = logging_config
        from swift.dev.recipe.tracking import RunTracker
        self.tracker = RunTracker(logging_config, output_dir)
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.no_save_optim = no_save_optim
        self.no_save_rng = no_save_rng
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        self.save_total_limit = save_total_limit
        self.manual_gc = manual_gc
        self.manual_gc_steps = manual_gc_steps
        if self.manual_gc_steps < 0:
            raise ValueError('manual_gc_steps must be >= 0.')
        self.num_generations = rlhf_config.num_generations
        self.num_ppo_epochs = max(1, rlhf_config.num_ppo_epochs)
        from twinkle.advantage import GAEAdvantage
        # gamma/lam are GAE *construction* hyperparameters (its __call__ is keyword-only for masks/
        # normalize), so they are bound here rather than passed per call.
        self._gae = GAEAdvantage(gamma=rlhf_config.gamma, gae_lambda=rlhf_config.lam)
        from swift.dev.recipe.train_loop import PromptBatchScheduler
        #: Prompt-set iterator (B1): one generation batch per rollout, exhausted after ``num_train_epochs``
        #: passes. PPO counts one rollout as one step (``_record_step`` per rollout), so the recipe sizes
        #: ``max_steps`` to ``len(self._prompt_batches)`` rather than to an optimizer-step budget.
        self._prompt_batches = PromptBatchScheduler(
            len(prompts),
            generation_batch_size=generation_batch_size,
            num_train_epochs=num_train_epochs,
            seed=seed)
        self.global_step = 0
        self.history: list = []

    def _score_rewards(self, samples: List[Any]) -> List[float]:
        """Scalar reward per sample: mean of the reward model(s)' seq_cls score (0 if none)."""
        if not self.reward_models:
            return [0.0] * len(samples)
        features = [s.input_feature for s in samples]
        totals = [0.0] * len(samples)
        for rm in self.reward_models:
            scores = rm.forward_only(inputs=features, return_logits=True)['logits'].reshape(-1).tolist()
            totals = [t + float(s) for t, s in zip(totals, scores)]
        return [t / len(self.reward_models) for t in totals]

    @staticmethod
    def _response_positions(feature: dict) -> List[int]:
        """Indices of the response tokens (labels != ignore) -- where logps and values are read."""
        labels = feature.get('labels') if isinstance(feature, dict) else None
        if labels is None:
            return []
        return [i for i, label in enumerate(labels) if label != _IGNORE_INDEX]

    def _token_values(self, sample: Any, positions: List[int]) -> List[float]:
        """The critic's per-response-token value ``V(s_t)`` at rollout time (the GAE input + clip anchor)."""
        out = self.value_model.forward_only(inputs=[sample.input_feature], return_logits=True, task='value')
        values = out.get('logits') if isinstance(out, dict) else None
        if values is None:
            raise RuntimeError("value model forward returned no logits; forward the critic with task='value'.")
        values = values.reshape(-1)
        return [float(values[p]) for p in positions]

    def _ref_logps_tokens(self, sample: Any, positions: List[int]) -> Optional[List[float]]:
        """Per-response-token reference log-probs for the KL penalty (disable_lora, or a frozen ref)."""
        if self.reference == 'disable_lora':
            out = self.model.forward_only(inputs=[sample.input_feature], disable_lora=True)
        else:
            out = self.reference.forward_only(inputs=[sample.input_feature])
        logps = out.get('logps') if isinstance(out, dict) else None
        if logps is None:
            return None
        logps = logps.reshape(-1)
        return [float(logps[p]) for p in positions]

    def _token_rewards(self, sample: Any, ref_tokens: Optional[List[float]], rm_score: float) -> List[float]:
        """Per-token reward: ``-kl_coef * (logp_t - ref_logp_t)`` with the RM scalar at the final token."""
        old = [float(x) for x in sample.old_logps]
        n = len(old)
        if n == 0:
            return []
        if ref_tokens is None:
            rewards = [0.0] * n
        else:
            rewards = [-self.rlhf_config.kl_coef * (old[t] - ref_tokens[t]) for t in range(n)]
        rewards[-1] += rm_score
        return rewards

    @staticmethod
    def _whiten_advantages(plans: List[list]) -> None:
        """Standardise per-token advantages across the whole rollout in place (TRL's reward whitening)."""
        import statistics
        flat = [a for plan in plans for a in plan[1]]
        if len(flat) < 2:
            return
        mean = statistics.fmean(flat)
        std = statistics.pstdev(flat) or 1.0
        for plan in plans:
            plan[1] = [(a - mean) / (std + 1e-8) for a in plan[1]]

    def _plan_rollout(self, samples: List[Any]) -> tuple:
        """Turn one rollout into per-sample ``[sample, advantages, returns, old_values]`` + mean reward.

        Everything the epochs re-use (advantages, returns, old_values) is computed here ONCE, before any
        optimizer step, so the ``num_ppo_epochs`` passes see the SAME anchors -- the definition of PPO's
        batch re-use. Samples with no response tokens are dropped.
        """
        cfg = self.rlhf_config
        rm_scores = self._score_rewards(samples)
        plans: List[list] = []
        total_reward = 0.0
        for sample, rm in zip(samples, rm_scores):
            positions = self._response_positions(sample.input_feature)
            if not positions:
                continue
            values = self._token_values(sample, positions)
            ref_tokens = self._ref_logps_tokens(sample, positions)
            rewards = self._token_rewards(sample, ref_tokens, rm)
            advantages, returns = self._gae(rewards, values)
            # GAE returns [1, T] tensors; flatten to the same per-token float lists the rest of the plan
            # carries (values/rewards), so the rollout anchors stay plain Python across num_ppo_epochs.
            advantages = advantages.squeeze(0).tolist()
            returns = returns.squeeze(0).tolist()
            plans.append([sample, advantages, returns, values])
            total_reward += sum(rewards)
        if cfg.whiten_rewards:
            self._whiten_advantages(plans)
        return plans, (total_reward / max(1, len(plans)))

    def _plan_mini_batches(self, plans: List[list]) -> List[List[list]]:
        """Split one rollout's plans into full ``train_batch_size`` chunks, warning on a dropped tail.

        Mirrors :meth:`GRPOLoop._plan_mini_batches` (reusing the same structure-agnostic
        ``split_mini_batches``): every ``forward_backward`` must carry exactly ``train_batch_size`` rows so
        slice_dp hands each DP rank ``per_device_train_batch_size`` of them. A rollout narrower than one
        mini-batch is a config error (the rollout width is generation_batch_size * num_generations, the same
        every round), so it raises rather than silently training nothing.
        """
        from swift.dev.recipe.grpo import split_mini_batches

        mini_batches = split_mini_batches(plans, self.train_batch_size)
        if not mini_batches:
            raise RuntimeError(
                f'rollout produced {len(plans)} trainable samples but train_batch_size={self.train_batch_size} '
                '(per_device_train_batch_size * dp_size), so no full mini-batch fits and no optimizer step is '
                'possible. Enlarge generation_batch_size / num_generations or the prompt set, or lower '
                'per_device_train_batch_size / the DP world size.')
        dropped = len(plans) - len(mini_batches) * self.train_batch_size
        if dropped:
            logger.warning('Dropping %d of %d rollout samples that do not fill a train_batch_size=%d mini-batch.',
                           dropped, len(plans), self.train_batch_size)
        return mini_batches

    def fit(self) -> list:
        """Run PPO over the prompt set for ``num_train_epochs`` passes (rollout -> GAE -> num_ppo_epochs).

        The prompt-set scheduler yields one generation batch per rollout and is exhausted after
        ``num_train_epochs`` passes (B1); an explicit ``max_steps`` still caps the rollout count early.
        Each rollout: per-token GAE (computed once, so the epochs re-use the SAME anchors), then the plans
        are split into ``train_batch_size``-row mini-batches (design B, mirroring GRPO) and replayed
        ``num_ppo_epochs`` times -- one policy ``forward_backward`` and one critic ``forward_backward`` per
        mini-batch, ``gradient_accumulation_steps`` mini-batches making one optimizer step. Feeding whole
        mini-batches (not one sample per step) is what makes DP>1 work: slice_dp needs >= dp_size rows per
        forward_backward or a rank gets no data. PPO still counts one rollout as one recorded step.
        """
        from swift.dev.recipe.train_loop import finish_manual_gc, start_manual_gc

        ga = self.gradient_accumulation_steps
        gc_was_enabled = start_manual_gc(self.manual_gc)
        try:
            for prompt_indices in self._prompt_batches:
                if self.max_steps > 0 and self.global_step >= self.max_steps:
                    break
                self.rollout.sync_weights()
                samples = self.rollout.generate(
                    [self.prompts[i] for i in prompt_indices],
                    num_samples=self.num_generations,
                    sampling_params=self.sampling_params)
                self.rollout.finish_generate()

                plans, mean_reward = self._plan_rollout(samples)
                mini_batches = self._plan_mini_batches(plans)
                for _ in range(self.num_ppo_epochs):
                    for mini_batch in mini_batches:
                        # Parallel lists over the mini-batch's rows: one entry per rollout sample, so
                        # slice_dp splits inputs and every advantage/return/logp/value column consistently.
                        inputs = [plan[0].input_feature for plan in mini_batch]
                        self.model.forward_backward(
                            inputs=inputs,
                            gradient_accumulation_steps=ga,
                            advantages=[plan[1] for plan in mini_batch],
                            old_logps=[plan[0].old_logps for plan in mini_batch])
                        self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                        self.value_model.forward_backward(
                            inputs=inputs,
                            gradient_accumulation_steps=ga,
                            task='value',
                            returns=[plan[2] for plan in mini_batch],
                            old_values=[plan[3] for plan in mini_batch])
                        self.value_model.clip_grad_and_step(
                            max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                self._record_step(mean_reward)
            return self.history
        finally:
            finish_manual_gc(gc_was_enabled)
            self.tracker.close()

    def _record_step(self, mean_reward: float) -> None:
        from swift.dev.recipe.train_loop import collect_manual_gc

        self.global_step += 1
        collect_manual_gc(self.manual_gc, self.manual_gc_steps, self.global_step)
        metrics = self.model.calculate_metric(is_training=True)
        value_metrics = self.value_model.calculate_metric(is_training=True)
        record = {
            'step': self.global_step,
            'loss': float(metrics['loss']) if metrics.get('loss') is not None else float('nan'),
            'value_loss': float(value_metrics['loss']) if value_metrics.get('loss') is not None else float('nan'),
            'reward': mean_reward,
        }
        record = self.tracker.log(record, self.global_step)
        self.history.append(record)
        if self.tracker.should_log(self.global_step):
            logger.info(f"step {self.global_step}  loss={record['loss']:.4f}  value_loss={record['value_loss']:.4f}  "
                        f"reward={record['reward']:.4f}")
        if self.save_steps and self.global_step % self.save_steps == 0:
            self.save(f'checkpoint-{self.global_step}')

    def save(self, name: str = 'checkpoint-final', *, is_final: bool = False) -> str:
        """Persist policy and critic checkpoints under one RL checkpoint directory.

        ``is_final`` is accepted for the shared loop.save contract (``TrainAssembly.save_final`` passes it
        to gate hub-push 'end'); this loop carries no hub_pusher, so it is inert here.
        """
        from swift.dev.recipe.train_loop import save_training_checkpoint

        save_training_checkpoint(
            self.model,
            name,
            output_dir=self.output_dir,
            consumed_train_samples=self.global_step,
            no_save_optim=self.no_save_optim,
            no_save_rng=self.no_save_rng,
            safe_serialization=self.safe_serialization,
            max_shard_size=self.max_shard_size,
            save_total_limit=self.save_total_limit)
        checkpoint_dir = os.path.join(self.output_dir, name)
        save_training_checkpoint(
            self.value_model,
            'value_model',
            output_dir=checkpoint_dir,
            consumed_train_samples=self.global_step,
            no_save_optim=self.no_save_optim,
            no_save_rng=self.no_save_rng,
            safe_serialization=self.safe_serialization,
            max_shard_size=self.max_shard_size,
            save_total_limit=self.save_total_limit)
        return checkpoint_dir

    def resume(self, state: dict, *, value_state: Optional[dict] = None) -> None:
        """Resume both optimizer phases and the completed rollout-step count."""
        self.global_step = int(state.get('consumed_train_samples', 0))
        if value_state is not None and int(value_state.get('consumed_train_samples', 0)) != self.global_step:
            raise ValueError('PPO policy and value checkpoints contain different completed-step counts.')
