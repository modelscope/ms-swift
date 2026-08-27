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
  - rollout = the same weight-syncable ``SamplerRollout`` (colocate / heterogeneous) run_grpo uses.

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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        ModelConfig,
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
    *,
    engine_args: Optional[Dict[str, Any]] = None,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run online PPO (policy + critic + reference + reward model + weight-syncable rollout).

    Reuses run_grpo's device planning and rollout, then trains the policy (clipped surrogate) and the
    critic (clipped value loss) over each rollout for ``num_ppo_epochs``. Returns the loss history.
    """
    from swift.dev.adapter import apply_tuner
    from swift.dev.builders import build_model, build_sampler, build_template
    from swift.dev.config import validate_configs
    from swift.dev.loss import configure_ppo_value_loss, configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe.run_grpo import (
        SamplerRollout,
        _grpo_sampling_params,
        _initialize_twinkle_rl,
        _prompts_from_dataset,
        _sampler_engine_args,
        plan_rl_device_groups,
    )
    from swift.model import get_model_processor

    if rlhf_config.rlhf_type != 'ppo':
        raise ValueError(f'run_ppo requires rlhf_type="ppo", got {rlhf_config.rlhf_type!r}.')
    validate_configs(model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
                     tuner_config, rlhf_config=rlhf_config)

    sampler_world_size = rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_data_parallel_size
    groups, sampler_remote_group, colocate = plan_rl_device_groups(distributed_config.nproc_per_node,
                                                                   rollout_config.vllm_mode, sampler_world_size)
    _initialize_twinkle_rl(distributed_config, groups)

    _, processor = get_model_processor(model_config.model, load_model=False)
    template = build_template(template_config, processor)
    ga = train_config.gradient_accumulation_steps

    # Policy (trainer): the clipped surrogate is the PPO policy loss (configure_rlhf_loss maps ppo).
    model = build_model(model_config, distributed_config, train_config, tuner_config)
    if tuner_config is not None:
        apply_tuner(model, tuner_config, gradient_accumulation_steps=ga)
    model.set_processor(InputProcessor, padding_free=template_config.padding_free)
    model.set_template(template)
    configure_rlhf_loss(model, rlhf_config)
    configure_optimizer(model, train_config, num_training_steps=train_config.max_steps or 1)

    # Critic: a trainable seq_cls (num_labels=1) value model, trained by the clipped value loss.
    value_model = _build_value_model(model_config, rlhf_config, distributed_config, train_config, template)
    configure_ppo_value_loss(value_model, rlhf_config)
    configure_optimizer(value_model, train_config, num_training_steps=train_config.max_steps or 1)

    # Rollout (weight-syncable, exactly as run_grpo), reward model(s) and reference for the KL penalty.
    sampler = build_sampler(
        model_config,
        backend='vllm',
        engine_args=_sampler_engine_args(rollout_config, engine_args, colocate),
        template=template,
        remote_group=sampler_remote_group)
    rollout = SamplerRollout(model, sampler, colocate=colocate)
    reference = _build_reference(model_config, tuner_config, template)
    reward_models = _build_reward_models(rlhf_config, template)
    prompts = _prompts_from_dataset(dataset_config)

    loop = PPOLoop(
        model,
        value_model,
        rollout,
        reference,
        reward_models,
        prompts,
        rlhf_config=rlhf_config,
        max_steps=train_config.max_steps or 1,
        gradient_accumulation_steps=ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        sampling_params=_grpo_sampling_params(rlhf_config, generation_config))
    try:
        history = loop.fit()
    finally:
        rollout.shutdown()
    del output_dir, _save_final  # policy checkpointing of the RL run is a follow-up; smoke returns history
    return history


def _build_value_model(model_config: ModelConfig, rlhf_config: RLHFConfig, distributed_config: DistributedConfig,
                       train_config: TrainConfig, template: Any) -> Any:
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
    value_cfg.model = init_from
    value_cfg.task_type = 'seq_cls'  # the per-token value rides the seq_cls score head
    value_cfg.num_labels = 1
    value_model = build_model(value_cfg, distributed_config, train_config)
    value_model.set_processor(InputProcessor)
    value_model.set_template(template)
    return value_model


def _build_reference(model_config: ModelConfig, tuner_config: Optional[TunerConfig], template: Any) -> Any:
    """PPO's frozen reference for the KL penalty: 'disable_lora' (LoRA) or a frozen policy-init model.

    LoRA reuses the adapter-disabled base (no second model); full fine-tuning loads a frozen copy of
    the policy's initial weights (PPO anchors the KL to the starting policy).
    """
    if tuner_config is not None:
        return 'disable_lora'
    from swift.dev.builders import build_model
    from swift.dev.config import DistributedConfig
    from swift.dev.processor import InputProcessor

    ref = build_model(model_config, DistributedConfig(mode='local'))
    ref.set_processor(InputProcessor)
    ref.set_template(template)
    return ref


def _build_reward_models(rlhf_config: RLHFConfig, template: Any) -> List[Any]:
    """Build the frozen reward model(s) that score each completion (seq_cls, num_labels=1 scalar).

    Returns an empty list when none are configured -- the loop then relies solely on the KL/base reward
    and flags it. Reward models are frozen seq_cls scorers, forward_only'd for a scalar per sequence.
    """
    if not rlhf_config.reward_model:
        return []
    from swift.dev.builders import build_model
    from swift.dev.config import DistributedConfig, ModelConfig
    from swift.dev.processor import InputProcessor

    models: List[Any] = []
    for idx, rm_id in enumerate(rlhf_config.reward_model):
        rm_cfg = ModelConfig(model=rm_id, task_type='seq_cls')
        rm_cfg.num_labels = 1
        if rlhf_config.reward_model_type:
            rm_cfg.model_type = rlhf_config.reward_model_type[idx] if idx < len(rlhf_config.reward_model_type) else None
        rm = build_model(rm_cfg, DistributedConfig(mode='local'))
        rm.set_processor(InputProcessor)
        rm.set_template(template)
        models.append(rm)
    return models


class PPOLoop:
    """Per-token PPO loop: rollout -> per-token reward (KL + RM) -> GAE -> clipped policy + value.

    Each step rolls out completions from the weight-synced sampler, then for every response token builds
    a reward ``r_t = -kl_coef * (logp_t - ref_logp_t)`` (the reward model's scalar added at the final
    token) and runs :func:`compute_gae` over the response to get per-token advantages and returns. The
    policy is updated by the clipped surrogate (``GRPOLoss`` over per-token ``advantages``/``old_logps``)
    and the critic by the clipped value loss (``PPOValueLoss`` over per-token ``returns``/``old_values``).
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
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        sampling_params: Optional[dict] = None,
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
        self.max_grad_norm = max_grad_norm
        self.sampling_params = sampling_params
        self.num_generations = rlhf_config.num_generations
        self.num_ppo_epochs = max(1, rlhf_config.num_ppo_epochs)
        from twinkle.advantage import GAEAdvantage
        self._gae = GAEAdvantage()
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
            advantages, returns = self._gae(rewards, values, cfg.gamma, cfg.lam)
            plans.append([sample, advantages, returns, values])
            total_reward += sum(rewards)
        if cfg.whiten_rewards:
            self._whiten_advantages(plans)
        return plans, (total_reward / max(1, len(plans)))

    def fit(self) -> list:
        """Run max_steps PPO steps (each: rollout -> per-token GAE -> num_ppo_epochs of policy+critic)."""
        ga = self.gradient_accumulation_steps
        for _ in range(self.max_steps):
            if hasattr(self.rollout, 'sync_weights'):
                self.rollout.sync_weights()
            samples = self.rollout.generate(
                self.prompts, num_samples=self.num_generations, sampling_params=self.sampling_params)
            if hasattr(self.rollout, 'finish_generate'):
                self.rollout.finish_generate()

            plans, mean_reward = self._plan_rollout(samples)
            for _ in range(self.num_ppo_epochs):
                for sample, advantages, returns, old_values in plans:
                    inputs = [sample.input_feature]
                    self.model.forward_backward(
                        inputs=inputs, gradient_accumulation_steps=ga,
                        advantages=[advantages], old_logps=[sample.old_logps])
                    self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                    self.value_model.forward_backward(
                        inputs=inputs, gradient_accumulation_steps=ga, task='value',
                        returns=[returns], old_values=[old_values])
                    self.value_model.clip_grad_and_step(
                        max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
            self._record_step(mean_reward)
        return self.history

    def _record_step(self, mean_reward: float) -> None:
        self.global_step += 1
        metrics = self.model.calculate_metric(is_training=True)
        value_metrics = self.value_model.calculate_metric(is_training=True)
        record = {
            'step': self.global_step,
            'loss': float(metrics['loss']) if metrics.get('loss') is not None else float('nan'),
            'value_loss': float(value_metrics['loss']) if value_metrics.get('loss') is not None else float('nan'),
            'reward': mean_reward,
        }
        self.history.append(record)
        logger.info(f"step {self.global_step}  loss={record['loss']:.4f}  value_loss={record['value_loss']:.4f}  "
                    f"reward={record['reward']:.4f}")
