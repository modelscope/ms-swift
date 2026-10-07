"""Offline preference assembly: run_dpo orchestration (dpo / cpo / orpo / simpo / kto / rm).

Peer of ``run_sft`` for the OFFLINE preference family -- the RLHF types that need no rollout and no
weight-sync, only a fixed dataset of chosen/rejected pairs. One entry point dispatches on
``RLHFConfig.rlhf_type`` because the six share almost everything (data pipeline, loop, save); they
differ only in (a) which twinkle loss ``configure_rlhf_loss`` picks and (b) whether a reference model
is consulted:

  - dpo / kto: consult a frozen reference. LoRA runs the base model with the adapter DISABLED
    (``forward_only(disable_lora=True)``) so no second copy is loaded; full fine-tuning loads a
    separate frozen ``ref_model`` (defaulted to the policy's init by process.py::_derive_rlhf_ref_model).
  - cpo / orpo / simpo: reference-FREE -- the loss builds its own baseline, so no reference forward.
  - rm: a ``task_type='seq_cls', num_labels=1`` reward head scored pairwise by RewardLoss; no logps,
    no reference.

Data pipeline (Subsystem B): the swift template encodes a preference row into ``chosen_*`` / ``rejected_*``
fields (template ``mode='rlhf'``, or ``'kto'`` for kto; RM rides ``task_type='seq_cls'`` which drops the
labels). For the PAIRED types (dpo/cpo/orpo/simpo/rm) :class:`PreferenceLoop` splits each row into two
InputFeatures and feeds them INTERLEAVED -- ``[chosen_1, rejected_1, chosen_2, rejected_2, ...]`` -- which
is exactly the layout the twinkle DPO family's ``_split_chosen_rejected`` (even/odd indices) expects, and it
keeps every micro-batch at an even, equal sequence count so gradient accumulation stays correct.

KTO is the exception: it is UNPAIRED. Each row is a single completion carrying a binary ``label``
(desirable/undesirable), and the loss (twinkle ``KTOLoss``) anchors the policy-vs-reference log-ratio on a
detached reference point ``z_KL`` estimated over a mismatched "KL batch" (each prompt paired with an
unrelated completion). PreferenceLoop builds that KL batch the way legacy's KTOPreprocessor does -- rotate
the completions by one within the batch -- scores it with two no-grad forwards, and feeds the single
completions (not an interleave) to the policy forward with ``label``/``z_kl``/``ref_logps`` as loss kwargs.
``calculate_KL=False`` skips the KL forwards and anchors at z_KL=0.

Placement (RL_PLAN basic principle 2): the offline preference family runs ``mode='ray'`` like every other
RL path -- the policy is a Ray actor on the 'model' DeviceGroup, and a FULL-PARAMETER frozen reference (full
fine-tuning, or an explicit reference adapter) is a separate Ray actor on its own 'ref' group, built with
the run's backend (basic principle 1: an auxiliary only ``forward_only``s, which megatron and transformers
implement identically). A LoRA reference reuses the adapter-disabled policy base
(``forward_only(disable_lora=True)``) and plans no second group. The reference-logps path returns per-token
logps from the reference actor and hands them straight back into the policy forward.
"""
from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RLHFConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = logging.getLogger(__name__)

#: rlhf_types that consult a frozen reference model (the rest build their own baseline in the loss).
_REF_TYPES = frozenset({'dpo', 'kto'})
#: rlhf_types this recipe handles (ppo is online + needs a critic; grpo/gkd have their own recipes).
_OFFLINE_TYPES = frozenset({'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'})
#: The DeviceGroup name for a full-parameter frozen reference. Defined locally rather than imported from
#: the online run_grpo base because the offline family plans its OWN groups -- it has no sampler, so it does
#: not reuse plan_rl_device_groups -- and the name only has to agree between this module's initialize_twinkle
#: planning and its build_model(remote_group=...) below.
_REF_GROUP = 'ref'


def _reference_needs_group(rlhf_config: RLHFConfig, tuner_config: Optional[TunerConfig]) -> bool:
    """Whether _build_reference builds a separate frozen reference actor needing its own DeviceGroup.

    Mirrors ``_build_reference``'s branch exactly so the group PLANNING and the BUILD cannot drift: the
    reference-free types (cpo/orpo/simpo/rm), an explicit ``reference_free`` dpo/kto, and a LoRA run without
    a reference adapter (which reuses the adapter-disabled policy base, 'disable_lora') all need NO group;
    only a full-parameter frozen copy -- full fine-tuning, or an explicit reference adapter -- is a separate
    Ray actor on its own GPUs (RL_PLAN §3.5: full-parameter auxiliaries are heterogeneous, LoRA ones share).
    """
    if rlhf_config.rlhf_type not in _REF_TYPES or rlhf_config.reference_free:
        return False
    return not (tuner_config is not None and not rlhf_config.ref_adapters)


def run_dpo(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rlhf_config: RLHFConfig,
    tuner_config: Optional[TunerConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run an offline preference optimisation from atomic Configs. Returns loss history.

    Dispatches on ``rlhf_config.rlhf_type`` (one of dpo/kto/cpo/orpo/simpo/rm). The build order is
    :class:`~swift.dev.recipe.assembly.TrainAssembly`'s, shared with every other recipe; what this one
    owns is the preference data pipeline (raw rows -> chosen/rejected -> interleaved features), the
    optional reference-logps forward that dpo/kto add, and :class:`PreferenceLoop` in place of SFTLoop.
    Because the loop differs, the stages are driven one by one instead of through ``fit``.
    """
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly

    rlhf_type = rlhf_config.rlhf_type
    if rlhf_type not in _OFFLINE_TYPES:
        raise ValueError(f'run_dpo handles the offline preference types {sorted(_OFFLINE_TYPES)}, got '
                         f'rlhf_type={rlhf_type!r}. Use run_grpo (online RL), run_gkd (distillation) or '
                         'run_ppo instead.')
    assembly = TrainAssembly(
        'run_dpo',
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
    # A FULL-PARAMETER frozen reference is a separate Ray actor on its own 'ref' DeviceGroup (RL_PLAN
    # §3.5); a LoRA / reference-free run plans none. The name/world size here match what _build_reference
    # builds below, decided by the SAME predicate (_reference_needs_group) so plan and build cannot drift.
    ref_remote_group: Optional[str] = None
    auxiliary_groups: List[Tuple[str, int]] = []
    if _reference_needs_group(rlhf_config, tuner_config):
        ref_remote_group = _REF_GROUP
        auxiliary_groups.append((_REF_GROUP, 1))
    TrainAssembly.initialize_twinkle(
        distributed_config,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        auxiliary_groups=auxiliary_groups)

    assembly.build_template()
    # Encode with the preference template mode: 'kto' for kto (allows a missing rejected), else
    # 'rlhf'. RM additionally rides task_type='seq_cls', which makes encode drop the labels.
    assembly.template.set_mode('kto' if rlhf_type == 'kto' else 'rlhf')

    # Raw (un-encoded) rows: build_dataset(encode=False) keeps the preference columns intact and hands
    # back list[row] batches; PreferenceLoop encodes + interleaves each batch itself (the SFT encode
    # path only knows single-sequence causal_lm, not chosen/rejected pairs).
    assembly.build_dataset(encode=False)
    assembly.plan_steps()
    assembly.build_model()
    configure_rlhf_loss(assembly.model, rlhf_config)
    configure_optimizer(
        assembly.model,
        train_config,
        num_training_steps=assembly.total_opt_steps,
        distributed_config=distributed_config)

    # W24: periodic evaluation is not wired for the offline preference loop -- PreferenceLoop stores
    # eval_dataloader/eval_steps but fit() never evaluates, so a configured eval split would be silently
    # ignored. Fail loudly until the val-loss eval path lands, rather than train without the eval asked for.
    if assembly.eval_dataloader is not None and train_config.eval_steps:
        raise NotImplementedError(
            'Periodic evaluation is not implemented for offline preference training (dpo/kto/cpo/orpo/'
            'simpo/rm) yet, so an eval split would be silently ignored. Remove the eval dataset / '
            '--eval_steps for this run.')
    assembly.loop = PreferenceLoop(
        assembly.model,
        assembly.dataloader,
        assembly.template,
        rlhf_type=rlhf_type,
        reference=_build_reference(
            assembly.model, rlhf_config, tuner_config, distributed_config, remote_group=ref_remote_group),
        calculate_kl=bool(rlhf_config.calculate_KL is not False),
        max_steps=assembly.total_opt_steps,
        num_train_epochs=train_config.num_train_epochs,
        gradient_accumulation_steps=assembly.ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config,
        output_dir=output_dir,
        eval_dataloader=assembly.eval_dataloader,
        eval_steps=train_config.eval_steps,
        save_steps=checkpoint_config.save_steps,
        no_save_optim=checkpoint_config.no_save_optim or checkpoint_config.save_only_model,
        no_save_rng=checkpoint_config.no_save_rng or checkpoint_config.save_only_model,
        safe_serialization=checkpoint_config.safe_serialization,
        max_shard_size=checkpoint_config.max_shard_size,
        save_total_limit=checkpoint_config.save_total_limit,
        ignore_data_skip=checkpoint_config.ignore_data_skip,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_eval=bool(megatron_config and megatron_config.manual_gc_eval),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0)
    if assembly.resume_dir:
        assembly.loop.resume(assembly.resume_model())
    history = assembly.loop.fit()
    if _save_final:
        assembly.save_final()
    return history


def _build_reference(model: TrainableModel,
                     rlhf_config: RLHFConfig,
                     tuner_config: Optional[TunerConfig],
                     distributed_config: DistributedConfig,
                     *,
                     remote_group: Optional[str]) -> Optional[Any]:
    """The reference the loop consults for ref-logps, or None for the reference-free types.

    Returns one of:
      - None: cpo/orpo/simpo/rm (reference-free), dpo/kto with ``reference_free=True``, OR dpo/kto under
        LoRA -- LoRA needs no object because the reference is the SAME model with the adapter disabled,
        which the loop reaches via ``forward_only(disable_lora=True)`` (signalled by the sentinel string
        'disable_lora').
      - 'disable_lora': dpo/kto + LoRA (adapter-disabled base is the reference).
      - a frozen TrainableModel: dpo/kto + full fine-tuning (a separate ref_model copy), a Ray actor on the
        ``remote_group`` DeviceGroup the caller planned for it.
    """
    if rlhf_config.rlhf_type not in _REF_TYPES:
        return None
    if rlhf_config.reference_free:
        # Reference-free DPO/KTO: DPOLoss.reference_free anchors on the policy alone, so no reference is
        # built and the loop passes no ref-logps. This is wired in lockstep with configure forwarding
        # ``reference_free`` to the loss -- with a None reference but reference_free=False, DPOLoss would
        # hit its zero-loss fallthrough and silently train on nothing.
        return None
    if tuner_config is not None and not rlhf_config.ref_adapters:
        # LoRA without an explicit reference adapter: the frozen base is the reference. It reuses the
        # policy's already-loaded base (no second copy), so it needs no group and no remote_group.
        return 'disable_lora'
    # Full fine-tuning, or an explicit reference adapter: load a separate frozen reference.
    return _load_frozen_reference(model, rlhf_config, distributed_config, remote_group=remote_group)


def _load_frozen_reference(model: TrainableModel,
                           rlhf_config: RLHFConfig,
                           distributed_config: DistributedConfig,
                           *,
                           remote_group: Optional[str]) -> Any:
    """Build a frozen reference model from rlhf_config.ref_model, sharing the policy's processor/template.

    Reuses the policy's InputProcessor + template so both models encode a batch identically (same
    padding, same shift), which is what lets the loop feed one interleaved feature list to both and
    line up the per-token logps. No optimizer/tuner: the reference is only ever forward_only'd.

    The full-parameter reference is an independent weight copy, so it is heterogeneous (RL_PLAN §3.5): a
    separate Ray actor on its own ``remote_group`` DeviceGroup, built with the run's backend (basic
    principle 1 -- ``forward_only`` is identical for megatron and transformers).
    """
    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.config import ModelConfig
    from swift.dev.recipe.assembly import configure_frozen_adapter

    if rlhf_config.ref_model is None:
        raise ValueError('dpo/kto full fine-tuning needs a reference model, but RLHFConfig.ref_model is None. '
                         'It is normally defaulted to the policy model by process.py::_derive_rlhf_ref_model; '
                         'pass --ref_model explicitly if you bypassed config processing.')
    if remote_group is None:
        raise ValueError('A full-parameter frozen reference must be built on its own DeviceGroup, but '
                         'remote_group is None. _reference_needs_group planned no group for it, which '
                         'contradicts reaching _load_frozen_reference -- the plan and the build have drifted.')
    # A bare ModelConfig pointing at ref_model, no tuner and no optimizer: the reference is only ever
    # forward_only'd. Its DistributedConfig inherits the run backend but is forced to ray on world_size=1.
    ref_cfg = ModelConfig(model=rlhf_config.ref_model)
    ref_cfg.model_type = rlhf_config.ref_model_type
    ref_cfg.model_revision = rlhf_config.ref_model_revision
    ref = build_model(
        ref_cfg, frozen_auxiliary_distributed_config(distributed_config, 1), remote_group=remote_group)
    return configure_frozen_adapter(
        ref,
        model.template if hasattr(model, 'template') else None,
        rlhf_config.ref_adapters,
        role='ref')


class PreferenceLoop:
    """Offline preference training loop: interleave chosen/rejected features, forward_backward, step.

    Peer of :class:`SFTLoop` for the preference family. Per micro-batch it encodes the raw preference
    rows, lays the features out interleaved (``[chosen_1, rejected_1, ...]``), optionally runs a
    reference forward for ref-logps (dpo/kto), then a single policy forward_backward with those
    ref-logps as a loss kwarg -- the same one-micro-batch-per-step GA shape SFTLoop uses on the
    transformers backend, so the grad-sync gate lines up.
    """

    def __init__(
        self,
        model: TrainableModel,
        dataloader: Any,
        template: Any,
        *,
        rlhf_type: str,
        reference: Optional[Any] = None,
        calculate_kl: bool = True,
        max_steps: int = -1,
        num_train_epochs: float = 1.0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 1,
        logging_config: Optional['LoggingConfig'] = None,
        save_steps: Optional[int] = None,
        output_dir: str = 'output',
        eval_dataloader: Any = None,
        eval_steps: Optional[int] = None,
        no_save_optim: bool = False,
        no_save_rng: bool = False,
        safe_serialization: bool = True,
        max_shard_size: str = '5GB',
        save_total_limit: Optional[int] = None,
        ignore_data_skip: bool = False,
        manual_gc: bool = False,
        manual_gc_eval: bool = True,
        manual_gc_steps: int = 0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.template = template
        self.rlhf_type = rlhf_type
        self.reference = reference
        #: KTO only: whether to estimate the z_KL reference point from a mismatched KL batch. Off
        #: (``calculate_KL=False``) anchors the KTO loss at z_KL=0 and skips the two extra KL forwards.
        self.calculate_kl = calculate_kl
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_config.logging_steps if logging_config is not None else logging_steps
        self.logging_config = logging_config
        from swift.dev.recipe.tracking import RunTracker
        self.tracker = RunTracker(logging_config, output_dir)
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.eval_dataloader = eval_dataloader
        self.eval_steps = eval_steps
        self.no_save_optim = no_save_optim
        self.no_save_rng = no_save_rng
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        self.save_total_limit = save_total_limit
        self.ignore_data_skip = ignore_data_skip
        self.manual_gc = manual_gc
        self.manual_gc_eval = manual_gc_eval
        self.manual_gc_steps = manual_gc_steps
        if self.manual_gc_steps < 0:
            raise ValueError('manual_gc_steps must be >= 0.')
        self._start_epoch = 0
        # RM scores a seq_cls head (no labels, no logps); the rest read per-token labels.
        self._is_reward = rlhf_type == 'rm'
        self.global_step = 0
        self.micro_step = 0
        self.history: list = []

    def _reached_max(self) -> bool:
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def _is_grad_sync_boundary(self) -> bool:
        """twinkle's do_grad_sync gate, loop-side: ga==1 every step, else one micro-step late."""
        ga = self.gradient_accumulation_steps
        return ga == 1 or ((self.micro_step - 1) % ga == 0 and self.micro_step > 1)

    @staticmethod
    def _strip_side(encoded: dict, prefix: str) -> dict:
        """Extract one ``<prefix>_*`` side of an encoded preference row into a standalone InputFeature.

        ``length`` is per-side bookkeeping the collator does not consume; drop it so the feature is just
        the model inputs (input_ids [+ labels/loss_scale]).
        """
        feature = {k[len(prefix) + 1:]: v for k, v in encoded.items() if k.startswith(prefix + '_')}
        feature.pop('length', None)
        return feature

    def _encode_pair(self, row: dict) -> List[dict]:
        """Encode one raw preference row into ``[chosen_feature, rejected_feature]``.

        The template (mode rlhf) yields a single dict with ``chosen_*`` / ``rejected_*`` keys; we strip
        the prefixes back into two standalone InputFeatures. RM (seq_cls) has no labels, so only
        input_ids survive -- RewardLoss scores the head's logits, not logps.
        """
        encoded = self.template.encode(row)
        pair = []
        for prefix in ('chosen', 'rejected'):
            feature = self._strip_side(encoded, prefix)
            if 'input_ids' not in feature:
                raise ValueError(f'preference row did not encode a {prefix!r} sequence (no {prefix}_input_ids). '
                                 f'rlhf_type={self.rlhf_type} needs paired chosen/rejected data.')
            pair.append(feature)
        return pair

    def _interleave(self, rows: List[dict]) -> List[dict]:
        """Raw rows -> interleaved feature list ``[chosen_1, rejected_1, chosen_2, rejected_2, ...]``."""
        features: List[dict] = []
        for row in rows:
            features.extend(self._encode_pair(row))
        return features

    @staticmethod
    def _last_assistant_content(row: dict) -> Any:
        """The completion of a KTO row: the content of its final (assistant) message."""
        messages = row.get('messages') or []
        if not messages or messages[-1].get('role') != 'assistant':
            raise ValueError('KTO rows must end with an assistant completion (last message '
                             f"role='assistant'); got messages={messages!r}.")
        return messages[-1]['content']

    def _encode_kto_batch(self, rows: List[dict]) -> Tuple[List[dict], List[dict], List[bool]]:
        """Encode a KTO batch into ``(completion_features, kl_features, labels)``.

        KTO is UNPAIRED: each row is a single completion with a binary ``label`` (desirable/undesirable),
        so there is no chosen/rejected interleave. The loss also needs a reference point z_KL estimated
        over a MISMATCHED "KL batch" -- each prompt paired with an unrelated completion. We build that
        batch the way legacy's KTOPreprocessor does: rotate the completions by one within the batch, so
        prompt_i is scored against completion_{i-1} (wraparound at the front). Setting ``rejected_response``
        reuses the template's kto encode, which returns ``chosen_*`` (the completion), ``rejected_*`` (the
        KL completion) and ``label`` in one pass.
        """
        completions = [self._last_assistant_content(row) for row in rows]
        rotated = [completions[-1]] + completions[:-1]
        completion_features: List[dict] = []
        kl_features: List[dict] = []
        labels: List[bool] = []
        for row, kl_completion in zip(rows, rotated):
            encoded = self.template.encode({**row, 'rejected_response': kl_completion})
            completion = self._strip_side(encoded, 'chosen')
            if 'input_ids' not in completion:
                raise ValueError('KTO row did not encode a completion (no chosen_input_ids).')
            completion_features.append(completion)
            kl_features.append(self._strip_side(encoded, 'rejected'))
            labels.append(bool(encoded['label']))
        return completion_features, kl_features, labels

    def _sequence_logps(self, logps: Any, features: List[dict]) -> Any:
        """Reduce per-token forward logps ``[n, seq_len]`` to one summed sequence logp per feature.

        ``forward_only`` returns full-sequence, right-padded logps in submission order, so each feature's
        trainable (response) tokens sit at ``response_positions`` and right-padding never shifts them;
        summing the row there gives ``log pi(y|x)`` for that completion (the same frame the loss mask uses).
        """
        import torch

        from swift.dev.recipe._teacher import response_positions
        values = torch.as_tensor(logps).detach().float()
        if values.dim() == 1:
            values = values.unsqueeze(0)
        seq = []
        for row, feature in zip(values, features):
            positions = response_positions(feature)
            row = row.reshape(-1)
            seq.append(row[positions].sum() if positions else row.new_zeros(()))
        return torch.stack(seq)

    def _kto_kl_term(self, kl_features: List[dict]) -> Optional[float]:
        """z_KL: the detached, clamped KL(pi||pi_ref) estimate over the mismatched KL batch.

        Mirrors TRL/legacy KTO: ``z_KL = mean(seq_logp_pi - seq_logp_ref).clamp(min=0)``, detached -- a
        reference POINT, never a gradient path. Both forwards are no-grad (``forward_only``): the policy
        with its adapter active, the reference via the same ``_ref_logps`` mechanism the completion batch
        uses. Returns None when ``calculate_KL`` is off, which KTOLoss reads as z_KL=0.
        """
        if not self.calculate_kl or not kl_features:
            return None
        policy_out = self.model.forward_only(inputs=kl_features)
        policy_logps = policy_out.get('logps') if isinstance(policy_out, dict) else None
        ref_logps = self._ref_logps(kl_features)
        if policy_logps is None or ref_logps is None:
            return None
        policy_seq = self._sequence_logps(policy_logps, kl_features)
        ref_seq = self._sequence_logps(ref_logps, kl_features)
        return float((policy_seq - ref_seq).mean().clamp(min=0).detach())

    def _kto_step_kwargs(self, rows: List[dict], kwargs: Dict[str, Any]) -> List[dict]:
        """Assemble one KTO micro-batch: fill ``kwargs`` with its loss inputs, return the completion features.

        KTO is unpaired, so the policy forward is over the single completions (not an interleave). The
        KTOLoss needs three extras, all threaded through ``kwargs``: the per-example binary ``label``
        (desirable/undesirable), the reference logps for the SAME completions (the log-ratio anchor), and
        the detached ``z_kl`` reference point estimated over the mismatched KL batch. z_KL is omitted when
        ``calculate_KL`` is off, which KTOLoss reads as z_KL=0.
        """
        completion_features, kl_features, labels = self._encode_kto_batch(rows)
        ref_logps = self._ref_logps(completion_features)
        if ref_logps is not None:
            kwargs['ref_logps'] = ref_logps
        z_kl = self._kto_kl_term(kl_features)
        if z_kl is not None:
            kwargs['z_kl'] = z_kl
        kwargs['label'] = labels
        return completion_features

    def _ref_logps(self, features: List[dict]) -> Optional[Any]:
        """Per-token reference logps for this batch, or None when the type is reference-free.

        LoRA reference == the policy with its adapter disabled (no second model); full fine-tuning
        reference == the separate frozen model. Both are read with forward_only, which fills
        ``outputs['logps']`` for a logps-consuming loss.
        """
        if self.reference is None:
            return None
        if self.reference == 'disable_lora':
            outputs = self.model.forward_only(inputs=features, disable_lora=True)
        else:
            outputs = self.reference.forward_only(inputs=features)
        return outputs.get('logps')

    def fit(self) -> list:
        """Run the preference loop; returns the per-optimizer-step loss history."""
        from swift.dev.recipe.train_loop import finish_manual_gc, start_manual_gc

        ga = self.gradient_accumulation_steps
        epochs = math.ceil(self.num_train_epochs) if self.max_steps <= 0 else 10**9
        gc_was_enabled = start_manual_gc(self.manual_gc)
        try:
            for epoch in range(self._start_epoch, epochs):
                if self._reached_max():
                    break
                if hasattr(self.dataloader, 'set_epoch'):
                    self.dataloader.set_epoch(epoch)
                for rows in self.dataloader:
                    self.micro_step += 1
                    kwargs: Dict[str, Any] = {'gradient_accumulation_steps': ga}
                    if self._is_reward:
                        # RM trains a num_labels=1 seq_cls score head, not a vocab LM head. The forward must
                        # be told task='seq_cls' so the backend pools the head's per-token output to one
                        # [B, 1] score per sequence (megatron does this in forward_step's seq_cls branch;
                        # transformers' *ForSequenceClassification pools internally). Without it the megatron
                        # forward defaults to task='causal_lm' and RewardLoss reads raw [B, T, 1] per-token
                        # scores -> garbage. Same explicit task the RM inference paths pass (reward.py /
                        # run_ppo.py), so both backends are driven identically (basic principle 1).
                        kwargs['task'] = 'seq_cls'
                    if self.rlhf_type == 'kto':
                        features = self._kto_step_kwargs(list(rows), kwargs)
                    else:
                        features = self._interleave(list(rows))
                        if not self._is_reward:
                            ref_logps = self._ref_logps(features)
                            if ref_logps is not None:
                                kwargs['ref_logps'] = ref_logps
                    self.model.forward_backward(inputs=features, **kwargs)
                    is_boundary = self._is_grad_sync_boundary()
                    self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                    if is_boundary:
                        self._record_step()
                        if self._reached_max():
                            break
                if self.history:
                    self.tracker.log(self.history[-1], self.global_step, epoch_end=True)
            return self.history
        finally:
            finish_manual_gc(gc_was_enabled)
            self.tracker.close()

    def _record_step(self) -> None:
        """Count one optimizer step + log / periodic save (mirrors SFTLoop._record_step)."""
        from swift.dev.recipe.train_loop import collect_manual_gc

        self.global_step += 1
        collect_manual_gc(self.manual_gc, self.manual_gc_steps, self.global_step)
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        record = {'step': self.global_step, 'loss': loss}
        if metrics.get('grad_norm') is not None:
            record['grad_norm'] = float(metrics['grad_norm'])
        record = self.tracker.log(record, self.global_step)
        self.history.append(record)
        should_log = (self.tracker.should_log(self.global_step) if self.logging_config is not None else
                      bool(self.logging_steps and self.global_step % self.logging_steps == 0))
        if should_log:
            gn = record.get('grad_norm')
            gn_str = f'  grad_norm={gn:.4f}' if gn is not None else ''
            logger.info(f'step {self.global_step}  loss={loss:.4f}{gn_str}')
        if self.save_steps and self.global_step % self.save_steps == 0:
            self.save(f'checkpoint-{self.global_step}')

    def save(self, name: str = 'checkpoint-final', *, is_final: bool = False) -> str:
        """Persist the policy + training state via twinkle's native save (the reference is not saved).

        ``is_final`` is accepted for the shared loop.save contract (``TrainAssembly.save_final`` passes it
        to gate hub-push 'end'); this loop carries no hub_pusher, so it is inert here.
        """
        from swift.dev.recipe.train_loop import save_training_checkpoint

        consumed = self._dataloader_state().get('consumed_train_samples', 0)
        return save_training_checkpoint(
            self.model,
            name,
            output_dir=self.output_dir,
            consumed_train_samples=consumed,
            no_save_optim=self.no_save_optim,
            no_save_rng=self.no_save_rng,
            safe_serialization=self.safe_serialization,
            max_shard_size=self.max_shard_size,
            save_total_limit=self.save_total_limit)

    def _dataloader_state(self) -> dict:
        get_state = getattr(self.dataloader, 'get_state', None)
        return get_state() if get_state is not None else {}

    def resume(self, state: dict) -> None:
        """Resume optimizer-step counters and the preference dataloader position."""
        from swift.dev.recipe.train_loop import num_optimizer_steps

        cur_step = int(state['cur_step'])
        self.micro_step = cur_step
        self.global_step = num_optimizer_steps(cur_step, self.gradient_accumulation_steps)
        if self.ignore_data_skip:
            self._start_epoch = 0
            return
        consumed = int(state.get('consumed_train_samples', 0))
        if hasattr(self.dataloader, 'skip_consumed_samples'):
            self.dataloader.skip_consumed_samples(consumed)
        self._start_epoch = self._dataloader_state().get('resume_epoch', 0)
