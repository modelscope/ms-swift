"""The offline preference family's batch encoder + console log line (DPO/KTO/CPO/ORPO/SimPO/RM).

``run_dpo`` drives the flat offline layer (:func:`swift.dev.recipe.offline_loop.run_offline`) like every
other offline recipe; the one leaf-specific piece it owns is how a batch of RAW preference rows becomes
``forward_backward`` / ``forward_only`` arguments. :class:`PreferenceBatchEncoder` is that piece: a callable
whose ``__call__(unit, *, owns_ga)`` is the ``micro_step_kwargs`` seam the flat layer threads through both the
training step and the val-loss eval. It replaces the old ``PreferenceLoop`` (an ``OfflineLoop`` leaf) -- the
loop scaffolding it inherited now lives in the flat ``StepRunner`` / ``Evaluator`` / ``CheckpointSaver``, so
only the preference data pipeline and the optional reference-logps forward remain here.

:func:`preference_log_line` renders the DPO reward margin/accuracy console line; ``run_dpo`` injects it as the
recorder's ``log_formatter`` so the flat, hook-free ``StepRunner`` still surfaces the preference signal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel

logger = get_logger()


class PreferenceBatchEncoder:
    """Encode raw preference rows into micro-batch loss kwargs, adding the reference forward dpo/kto need.

    Constructed by ``run_dpo`` with the policy ``model``, the preference ``template`` (mode 'rlhf', or 'kto'
    for KTO), the ``rl_type``, the resolved ``reference`` (a frozen model, the ``'disable_lora'`` sentinel, or
    None) and the KTO ``calculate_kl`` switch. Its ``__call__`` is passed to the flat offline layer as the
    ``micro_step_kwargs`` seam, so the SAME encoding feeds both the training forward and the val-loss eval --
    exactly as the old ``PreferenceLoop._micro_step_kwargs`` hook did, with no backend branch here (the flat
    layer's transformers/Megatron gradient-accumulation fork calls this one callable for both).
    """

    def __init__(self,
                 model: 'TrainableModel',
                 template: Any,
                 *,
                 rl_type: str,
                 reference: Optional[Any] = None,
                 calculate_kl: bool = True,
                 gradient_accumulation_steps: int = 1):
        self.model = model
        self.template = template
        self.rl_type = rl_type
        self.reference = reference
        #: KTO only: whether to estimate the z_KL reference point from a mismatched KL batch. Off
        #: (``calculate_KL=False``) anchors the KTO loss at z_KL=0 and skips the two extra KL forwards.
        self.calculate_kl = calculate_kl
        #: Added to the forward kwargs only when the backend does NOT own gradient accumulation
        #: (transformers), so twinkle's grad-sync gate steps the optimizer every ga-th micro-batch; Megatron
        #: derives GA from the microbatch-list length and must not receive it. Same value the assembly planned.
        self.gradient_accumulation_steps = gradient_accumulation_steps
        #: RM scores a num_labels=1 seq_cls head (no labels, no logps); the rest read per-token labels.
        self._is_reward = rl_type == 'rm'

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
                                 f'rl_type={self.rl_type} needs paired chosen/rejected data.')
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

        from swift.dev.recipe._scoring import response_positions
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
        policy_out = self.model.forward_only(inputs=kl_features, compute_loss=False)
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
            outputs = self.model.forward_only(inputs=features, disable_lora=True, compute_loss=False)
        else:
            outputs = self.reference.forward_only(inputs=features, compute_loss=False)
        return outputs.get('logps')

    def __call__(self, unit: Any, *, owns_ga: bool) -> Dict[str, Any]:
        """Encode one micro-batch of raw preference rows into ``forward_backward``/``forward_only`` args.

        ``unit`` is a single dataloader batch (transformers) or the ga batches of one Megatron optimizer
        window concatenated (Megatron); either way it is a flat iterable of raw rows, encoded here the same
        for both backends and for the val-loss eval path (the flat layer calls this one seam everywhere). KTO
        is unpaired -- its layout, per-example ``label`` and detached ``z_kl`` reference point are assembled
        by :meth:`_kto_step_kwargs`; the paired types interleave chosen/rejected and add the reference logps.
        RM rides ``task='seq_cls'`` so the backend pools the score head to one value per sequence.
        """
        rows = list(unit)
        kwargs: Dict[str, Any] = {}
        if self.rl_type == 'kto':
            kwargs['inputs'] = self._kto_step_kwargs(rows, kwargs)
        else:
            features = self._interleave(rows)
            if not self._is_reward:
                ref_logps = self._ref_logps(features)
                if ref_logps is not None:
                    kwargs['ref_logps'] = ref_logps
            kwargs['inputs'] = features
        if self._is_reward:
            kwargs['task'] = 'seq_cls'
        if not owns_ga:
            kwargs['gradient_accumulation_steps'] = self.gradient_accumulation_steps
        return kwargs


def preference_log_line(record: dict, loss: float) -> str:
    """Console head signal for the preference loops: loss plus the DPO reward margin and accuracy.

    The margin (chosen-reward minus rejected-reward) and the accuracy (fraction of pairs where chosen wins)
    are what show preference learning converging -- a rising margin and an accuracy climbing toward 100% --
    which the loss alone hides. Both come from the registered DPOMetric; they are absent for the unpaired kto
    and the seq_cls rm, so those fall back to loss + grad_norm. Injected as the flat ``StepRunner``'s
    ``log_formatter``; the ``on_log`` callback fan-out stays in the recorder, so this returns just the line.
    """
    parts = [f"step {record.get('step')}", f'loss={loss:.4f}']
    margins = record.get('rewards/margins')
    if margins is not None:
        parts.append(f'rewards/margins={margins:.4f}')
    accuracies = record.get('rewards/accuracies')
    if accuracies is not None:
        parts.append(f'rewards/accuracies={accuracies:.1f}%')
    gn = record.get('grad_norm')
    if gn is not None:
        parts.append(f'grad_norm={gn:.4f}')
    return '  '.join(parts)
