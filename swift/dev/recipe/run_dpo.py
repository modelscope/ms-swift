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
labels). :class:`PreferenceLoop` splits each row into two InputFeatures and feeds them INTERLEAVED --
``[chosen_1, rejected_1, chosen_2, rejected_2, ...]`` -- which is exactly the layout the twinkle DPO
family's ``_split_chosen_rejected`` (even/odd indices) expects, and it keeps every micro-batch at an
even, equal sequence count so gradient accumulation stays correct.

NOTE ON MODE: this is a single-process (mode='local') transformers recipe. The reference-logps path
returns per-token logps on the driver and hands them straight back into the policy forward, which the
in-process model supports directly; a Ray/Megatron preference variant is out of scope here.
"""
from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
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


def run_dpo(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rlhf_config: RLHFConfig,
    tuner_config: Optional[TunerConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run an offline preference optimisation from atomic Configs. Returns loss history.

    Dispatches on ``rlhf_config.rlhf_type`` (one of dpo/kto/cpo/orpo/simpo/rm). Mirrors run_sft's
    build order (template -> dataset -> model -> tuner -> loss -> optimizer -> loop), the difference
    being the preference data pipeline (chosen/rejected -> interleaved features) and the optional
    reference-logps forward that dpo/kto add.
    """
    from swift.dev.adapter import apply_tuner
    from swift.dev.builders import build_dataset, build_model, build_template
    from swift.dev.config import validate_configs
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe.run_sft import _initialize_twinkle, _write_ckpt_args_json
    from swift.dev.recipe.train_loop import num_optimizer_steps
    from swift.model import get_model_processor

    rlhf_type = rlhf_config.rlhf_type
    if rlhf_type not in _OFFLINE_TYPES:
        raise ValueError(f'run_dpo handles the offline preference types {sorted(_OFFLINE_TYPES)}, got '
                         f'rlhf_type={rlhf_type!r}. Use run_grpo (online RL), run_gkd (distillation) or '
                         'run_ppo instead.')
    validate_configs(model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
                     tuner_config, rlhf_config=rlhf_config)
    _initialize_twinkle(distributed_config)

    ga = train_config.gradient_accumulation_steps
    _, processor = get_model_processor(model_config.model, load_model=False)
    # Encode with the preference template mode: 'kto' for kto (allows a missing rejected), else
    # 'rlhf'. RM additionally rides task_type='seq_cls', which makes encode drop the labels.
    template = build_template(template_config, processor, task_type=model_config.task_type)
    template.set_mode('kto' if rlhf_type == 'kto' else 'rlhf')

    # Raw (un-encoded) rows: build_dataset(encode=False) keeps the preference columns intact and hands
    # back list[row] batches; PreferenceLoop encodes + interleaves each batch itself (the SFT encode
    # path only knows single-sequence causal_lm, not chosen/rejected pairs).
    dataloader, eval_dataloader = build_dataset(
        dataset_config,
        template,
        train_config,
        distributed_config=distributed_config,
        encode=False,
        template_config=template_config)

    total_opt_steps = _preference_step_budget(train_config, dataloader, ga, num_optimizer_steps)
    if total_opt_steps <= 0:
        raise ValueError(f'run_dpo: computed {total_opt_steps} optimizer steps -- the dataloader is too small for '
                         f'gradient_accumulation_steps={ga}, or it is streaming with no max_steps. Set '
                         'TrainConfig.max_steps explicitly, or provide more preference data.')

    model = build_model(model_config, distributed_config, train_config, tuner_config)
    if tuner_config is not None:
        apply_tuner(model, tuner_config, gradient_accumulation_steps=ga)
    model.set_processor(InputProcessor, padding_free=template_config.padding_free)
    model.set_template(template)
    configure_rlhf_loss(model, rlhf_config)
    configure_optimizer(model, train_config, num_training_steps=total_opt_steps)

    reference = _build_reference(model, rlhf_config, tuner_config)

    loop = PreferenceLoop(
        model,
        dataloader,
        template,
        rlhf_type=rlhf_type,
        reference=reference,
        max_steps=total_opt_steps,
        num_train_epochs=train_config.num_train_epochs,
        gradient_accumulation_steps=ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        output_dir=output_dir,
        eval_dataloader=eval_dataloader,
        eval_steps=train_config.eval_steps,
        save_steps=checkpoint_config.save_steps)
    history = loop.fit()
    if _save_final:
        import os
        loop.save('checkpoint-final')
        _write_ckpt_args_json(
            os.path.join(output_dir, 'checkpoint-final'), processor, model_config, template_config, tuner_config)
    return history


def _preference_step_budget(train_config: TrainConfig, dataloader: Any, ga: int, num_optimizer_steps) -> int:
    """Optimizer-step budget, mirroring run_sft: max_steps wins, else epochs * ceil(micro/ga)."""
    if train_config.max_steps and train_config.max_steps > 0:
        return train_config.max_steps
    try:
        micro_per_epoch = len(dataloader)
    except TypeError:
        return 0  # iterable/streaming: caller must set max_steps
    total_micro = math.ceil(micro_per_epoch * train_config.num_train_epochs)
    return num_optimizer_steps(total_micro, ga)


def _build_reference(model: TrainableModel, rlhf_config: RLHFConfig,
                     tuner_config: Optional[TunerConfig]) -> Optional[Any]:
    """The reference the loop consults for ref-logps, or None for the reference-free types.

    Returns one of:
      - None: cpo/orpo/simpo/rm (reference-free), OR dpo/kto under LoRA -- LoRA needs no object because
        the reference is the SAME model with the adapter disabled, which the loop reaches via
        ``forward_only(disable_lora=True)`` (signalled by returning the sentinel string 'disable_lora').
      - 'disable_lora': dpo/kto + LoRA (adapter-disabled base is the reference).
      - a frozen TrainableModel: dpo/kto + full fine-tuning (a separate ref_model copy).
    """
    if rlhf_config.rlhf_type not in _REF_TYPES:
        return None
    if tuner_config is not None:
        # LoRA: the frozen base (adapter off) is the reference; no second model is loaded.
        return 'disable_lora'
    # Full fine-tuning: a genuine frozen copy. process.py defaults ref_model to the policy's init.
    return _load_frozen_reference(model, rlhf_config)


def _load_frozen_reference(model: TrainableModel, rlhf_config: RLHFConfig) -> Any:
    """Build a frozen reference model from rlhf_config.ref_model, sharing the policy's processor/template.

    Reuses the policy's InputProcessor + template so both models encode a batch identically (same
    padding, same shift), which is what lets the loop feed one interleaved feature list to both and
    line up the per-token logps. No optimizer/tuner: the reference is only ever forward_only'd.
    """
    from swift.dev.builders import build_model
    from swift.dev.config import DistributedConfig, ModelConfig
    from swift.dev.processor import InputProcessor

    if rlhf_config.ref_model is None:
        raise ValueError('dpo/kto full fine-tuning needs a reference model, but RLHFConfig.ref_model is None. '
                         'It is normally defaulted to the policy model by process.py::_derive_rlhf_ref_model; '
                         'pass --ref_model explicitly if you bypassed config processing.')
    # A reference is always single-process local (it is only forward_only'd on the driver): a bare
    # ModelConfig pointing at ref_model, no tuner and no optimizer.
    ref_cfg = ModelConfig(model=rlhf_config.ref_model)
    ref_cfg.model_type = rlhf_config.ref_model_type
    ref_cfg.model_revision = rlhf_config.ref_model_revision
    ref = build_model(ref_cfg, DistributedConfig(mode='local'))
    ref.set_processor(InputProcessor)
    ref.set_template(model.template if hasattr(model, 'template') else None)
    return ref


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
        max_steps: int = -1,
        num_train_epochs: float = 1.0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 1,
        save_steps: Optional[int] = None,
        output_dir: str = 'output',
        eval_dataloader: Any = None,
        eval_steps: Optional[int] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.template = template
        self.rlhf_type = rlhf_type
        self.reference = reference
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.eval_dataloader = eval_dataloader
        self.eval_steps = eval_steps
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

    def _encode_pair(self, row: dict) -> List[dict]:
        """Encode one raw preference row into ``[chosen_feature, rejected_feature]``.

        The template (mode rlhf/kto) yields a single dict with ``chosen_*`` / ``rejected_*`` keys; we
        strip the prefixes back into two standalone InputFeatures. RM (seq_cls) has no labels, so only
        input_ids survive -- RewardLoss scores the head's logits, not logps.
        """
        encoded = self.template.encode(row)
        pair = []
        for prefix in ('chosen', 'rejected'):
            feature = {k[len(prefix) + 1:]: v for k, v in encoded.items() if k.startswith(prefix + '_')}
            # `length` is per-side bookkeeping the collator does not consume; drop it so the feature is
            # just the model inputs (input_ids [+ labels/loss_scale]).
            feature.pop('length', None)
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
        ga = self.gradient_accumulation_steps
        epochs = math.ceil(self.num_train_epochs) if self.max_steps <= 0 else 10**9
        for epoch in range(epochs):
            if self._reached_max():
                break
            if hasattr(self.dataloader, 'set_epoch'):
                self.dataloader.set_epoch(epoch)
            for rows in self.dataloader:
                self.micro_step += 1
                features = self._interleave(list(rows))
                kwargs: Dict[str, Any] = {'gradient_accumulation_steps': ga}
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
        return self.history

    def _record_step(self) -> None:
        """Count one optimizer step + log / periodic save (mirrors SFTLoop._record_step)."""
        self.global_step += 1
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        record = {'step': self.global_step, 'loss': loss}
        if metrics.get('grad_norm') is not None:
            record['grad_norm'] = float(metrics['grad_norm'])
        self.history.append(record)
        if self.logging_steps and self.global_step % self.logging_steps == 0:
            gn = record.get('grad_norm')
            gn_str = f'  grad_norm={gn:.4f}' if gn is not None else ''
            logger.info(f'step {self.global_step}  loss={loss:.4f}{gn_str}')
        if self.save_steps and self.global_step % self.save_steps == 0:
            self.save(f'checkpoint-{self.global_step}')

    def save(self, name: str = 'checkpoint-final') -> str:
        """Persist the policy + training state via twinkle's native save (the reference is not saved)."""
        consumed = getattr(self.dataloader, 'consumed_samples', 0)
        return self.model.save(name, output_dir=self.output_dir, save_optimizer=True, consumed_train_samples=consumed)
