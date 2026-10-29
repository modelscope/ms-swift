# Copyright (c) ModelScope Contributors. All rights reserved.
# The offline RLHF family's batch encoder + console line: raw rows -> forward kwargs.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel

logger = get_logger()


class PreferenceBatchEncoder:
    """Build one update's preference inputs and reference signals for training or evaluation.

    The shared loop calls this ``micro_step_fn`` once; the model owns micro-batch splitting.
    """

    def __init__(
        self,
        model: "TrainableModel",
        template: Any,
        *,
        rl_type: str,
        reference: Optional[Any] = None,
        calculate_kl: bool = True,
        micro_batch_size: int = 1,
        data_parallel_size: int = 1,
    ):
        self.model = model
        self.template = template
        self.rl_type = rl_type
        self.reference = reference
        #: KTO only: whether to estimate the z_KL reference point from a mismatched KL batch. Off
        #: (``calculate_KL=False``) anchors the KTO loss at z_KL=0 and skips the two extra KL forwards.
        self.calculate_kl = calculate_kl
        #: Cap the no-grad reference forwards as well as the trainable policy forward.
        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        #: RM scores a num_labels=1 seq_cls head (no labels, no logps); the rest read per-token labels.
        self._is_reward = rl_type == "rm"

    @staticmethod
    def _strip_side(encoded: dict, prefix: str) -> dict:
        """Extract one ``<prefix>_*`` side of an encoded preference row into a standalone InputFeature.

        ``length`` is per-side bookkeeping the collator does not consume; drop it so the feature is
        just the model inputs (input_ids [+ labels/loss_scale]).
        """
        feature = {k[len(prefix) + 1 :]: v for k, v in encoded.items() if k.startswith(prefix + "_")}
        feature.pop("length", None)
        return feature

    def _encode_pair(self, row: dict) -> List[dict]:
        """Encode one raw preference row into ``[chosen_feature, rejected_feature]``.

        The template (mode rlhf) yields a single dict with ``chosen_*`` / ``rejected_*`` keys; we
        strip the prefixes back into two standalone InputFeatures. RM (seq_cls) has no labels, so
        only input_ids survive -- RewardLoss scores the head's logits, not logps.
        """
        encoded = self.template.encode(row)
        pair = []
        for prefix in ("chosen", "rejected"):
            feature = self._strip_side(encoded, prefix)
            if "input_ids" not in feature:
                raise ValueError(
                    f"preference row did not encode a {prefix!r} sequence (no {prefix}_input_ids). "
                    f"rl_type={self.rl_type} needs paired chosen/rejected data."
                )
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
        messages = row.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(
                "KTO rows must end with an assistant completion (last message "
                f"role='assistant'); got messages={messages!r}."
            )
        return messages[-1]["content"]

    def _encode_kto_batch(self, rows: List[dict]) -> Tuple[List[dict], List[dict], List[bool]]:
        """Encode a KTO batch into ``(completion_features, kl_features, labels)``.

        KTO is UNPAIRED: each row is one completion with a binary ``label``. The loss also needs a
        reference point z_KL estimated over a MISMATCHED "KL batch" -- each prompt paired with an
        unrelated completion via rotating the batch's completions by one; ``rejected_response``
        reuses the template's kto encode (``chosen_*`` / ``rejected_*`` / ``label`` in one pass).
        """
        completions = [self._last_assistant_content(row) for row in rows]
        rotated = [completions[-1]] + completions[:-1]
        completion_features: List[dict] = []
        kl_features: List[dict] = []
        labels: List[bool] = []
        for row, kl_completion in zip(rows, rotated):
            encoded = self.template.encode({**row, "rejected_response": kl_completion})
            completion = self._strip_side(encoded, "chosen")
            if "input_ids" not in completion:
                raise ValueError("KTO row did not encode a completion (no chosen_input_ids).")
            completion_features.append(completion)
            kl_features.append(self._strip_side(encoded, "rejected"))
            labels.append(bool(encoded["label"]))
        return completion_features, kl_features, labels

    def _sequence_logps(self, logps: Any, features: List[dict]) -> Any:
        """Reduce per-token forward logps ``[n, seq_len]`` to one summed sequence logp per feature.

        ``forward_only`` returns full-sequence, right-padded logps in submission order, so each
        feature's trainable (response) tokens sit at ``response_positions`` and right-padding never
        shifts them; summing the row there gives ``log pi(y|x)`` for that completion (the same frame
        the loss mask uses).
        """
        import torch

        from swift.dev.utils.scoring import response_positions

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

        Mirrors TRL/legacy KTO: ``z_KL = mean(seq_logp_pi - seq_logp_ref).clamp(min=0)``, detached
        -- a reference POINT, never a gradient path. Both forwards are no-grad (``forward_only``);
        returns None when ``calculate_KL`` is off, which KTOLoss reads as z_KL=0.
        """
        if not self.calculate_kl or not kl_features:
            return None
        policy_out = self.model.forward_only(
            inputs=kl_features, compute_loss=False, micro_batch_size=self.micro_batch_size
        )
        policy_logps = policy_out.get("logps")
        ref_logps = self._ref_logps(kl_features)
        if policy_logps is None or ref_logps is None:
            return None
        policy_seq = self._sequence_logps(policy_logps, kl_features)
        ref_seq = self._sequence_logps(ref_logps, kl_features)
        return float((policy_seq - ref_seq).mean().clamp(min=0).detach())

    def _kto_step_kwargs(self, rows: List[dict], kwargs: Dict[str, Any]) -> List[dict]:
        """Assemble one KTO update: fill ``kwargs`` with its loss inputs, return the completions.

        The KTOLoss needs three extras threaded through ``kwargs``: the per-example binary ``label``,
        the reference logps for the SAME completions (the log-ratio anchor), and the detached
        ``z_kl`` reference point. z_KL is omitted when ``calculate_KL`` is off (KTOLoss reads 0).
        """
        completion_features, kl_features, labels = self._encode_kto_batch(rows)
        ref_logps = self._ref_logps(completion_features)
        if ref_logps is not None:
            kwargs["ref_logps"] = ref_logps
        z_kl = self._kto_kl_term(kl_features)
        if z_kl is not None:
            kwargs["z_kl"] = z_kl
        kwargs["label"] = labels
        return completion_features

    def _ref_logps(self, features: List[dict]) -> Optional[Any]:
        """Per-token reference logps for this batch, or None when the type is reference-free.

        The shared-weights LoRA view IS the policy with its adapter disabled (no second model); the
        full fine-tuning reference is the separate frozen model. Both are read with forward_only,
        which fills ``outputs['logps']`` for a logps-consuming loss.
        """
        if self.reference is None:
            return None
        if self.reference is self.model:
            outputs = self.model.forward_only(
                inputs=features,
                disable_lora=True,
                compute_loss=False,
                micro_batch_size=self.micro_batch_size * (1 if self.rl_type == "kto" else 2),
            )
        else:
            outputs = self.reference.forward_only(
                inputs=features,
                compute_loss=False,
                micro_batch_size=self.micro_batch_size * (1 if self.rl_type == "kto" else 2),
            )
        return outputs.get("logps")

    def __call__(self, unit: Any, *, micro_batch_size: int) -> Dict[str, Any]:
        """Encode one update, preserving paired sequences with a micro-batch size of 2B."""
        rows = list(unit)
        if self.rl_type != "kto" and len(rows) % self.data_parallel_size:
            raise ValueError(
                f"Preference updates need a record count divisible by DP={self.data_parallel_size}; "
                f"got {len(rows)} records. Each rank must receive whole chosen/rejected pairs."
            )
        kwargs: Dict[str, Any] = {}
        if self.rl_type == "kto":
            kwargs["inputs"] = self._kto_step_kwargs(rows, kwargs)
        else:
            features = self._interleave(rows)
            if not self._is_reward:
                ref_logps = self._ref_logps(features)
                if ref_logps is not None:
                    kwargs["ref_logps"] = ref_logps
            kwargs["inputs"] = features
        if self._is_reward:
            kwargs["task"] = "seq_cls"
        kwargs["gradient_accumulation_steps"] = 1
        kwargs["micro_batch_size"] = micro_batch_size * (1 if self.rl_type == "kto" else 2)
        return kwargs


def preference_log_line(record: dict, loss: float) -> str:
    """Console head signal for the preference loops: loss plus the DPO reward margin and accuracy.

    The margin (chosen-reward minus rejected-reward) and the accuracy (fraction of pairs where
    chosen wins) show preference learning converging, which the loss alone hides. Both come from the
    registered DPOMetric; they are absent for the unpaired kto and the seq_cls rm, so those fall
    back to loss + grad_norm.
    """
    parts = [f"step {record.get('step')}", f"loss={loss:.4f}"]
    margins = record.get("rewards/margins")
    if margins is not None:
        parts.append(f"rewards/margins={margins:.4f}")
    accuracies = record.get("rewards/accuracies")
    if accuracies is not None:
        parts.append(f"rewards/accuracies={accuracies:.1f}%")
    gn = record.get("grad_norm")
    if gn is not None:
        parts.append(f"grad_norm={gn:.4f}")
    return "  ".join(parts)
