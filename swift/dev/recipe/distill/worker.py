# Copyright (c) ModelScope Contributors. All rights reserved.
# Distillation's remote worker: the teacher forward and the per-update forward kwargs, in ONE actor.
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from twinkle import remote_class, remote_function
from twinkle_agentic.async_rl.score_worker import _run_grouped

from swift.dev.utils.scoring import encode_privileged_feature


class DistillRow(NamedTuple):
    """One distillation training row: the student feature plus the context a teacher view needs.

    Uniform across both ``lmbda`` sources -- an on-policy rollout sample and an off-policy dataset
    completion both reduce to ``(feature, messages, extra)`` -- so the loop's update grouping and
    the teacher scoring never branch on where a row came from.
    """

    feature: Dict[str, Any]
    messages: List[dict]
    extra: Dict[str, Any]


def response_ids_from_feature(feature: Dict[str, Any]) -> List[int]:
    """The response token ids of an encoded feature, recovered from its (next-token shifted) labels.

    Labels hold the response token ids at trainable positions and ``-100`` elsewhere, so filtering
    ``-100`` yields exactly the sampled response -- the tokens the teacher must re-score.
    """
    labels = feature.get("labels")
    if labels is None:
        raise ValueError("OPSD/MOPD teacher scoring requires a labelled feature to recover the response token ids.")
    response_ids = [int(label) for label in labels if int(label) != -100]
    if not response_ids:
        raise ValueError("OPSD/MOPD feature contains no response tokens to score.")
    return response_ids


@remote_class(execute="first")
class DistillWorker:
    """Build one update's ``forward_backward`` kwargs where the teacher signal is produced.

    The teacher forward runs HERE (a CPU actor) so its heavy output never materialises in the
    driver: GKD's full-vocab logits and OPSD / MOPD's response-only logps are declared ``stash``
    channels and ride the kwargs as refs Ray derefs at the training actor. ``teachers`` is
    always a list of bare frozen models, described by the plain parameters below.
    """

    def __init__(
        self,
        *,
        teachers: Sequence[Any],
        signal: str,
        template: Any = None,
        disable_lora: bool = False,
        offload: bool = False,
        placements: Optional[Sequence[str]] = None,
        gkd_logits_topk: Optional[int] = None,
        sft_alpha: float = 0.0,
        teacher_weights: Optional[Sequence[float]] = None,
        train_batch_size: int = 1,
    ):
        if signal not in ("logits", "logps", "multi_logps"):
            raise ValueError(f"distillation signal must be one of 'logits'/'logps'/'multi_logps', got {signal!r}.")
        self.teachers = list(teachers)
        self.signal = signal
        self.template = template
        self.disable_lora = disable_lora
        self.offload = offload
        self.placements = None if placements is None else list(placements)
        self.gkd_logits_topk = gkd_logits_topk
        self.sft_alpha = sft_alpha
        self.teacher_weights = None if teacher_weights is None else list(teacher_weights)
        self.train_batch_size = max(1, train_batch_size)

    def _response_logps(self, scorer: Any, features: List[dict], *, disable_lora: bool = False) -> List[List[float]]:
        """Score ``features`` with ``forward_only`` -> one response-token logp list per feature.

        The features form one update (``forward_kwargs`` builds its signals together), so one
        whole-batch call -- the dispatch splits it per rank -- is the whole story.
        """
        from swift.dev.utils.scoring import response_token_scores

        frame = scorer.forward_only(
            inputs=features,
            micro_batch_size=max(1, self.train_batch_size // scorer.device_mesh.data_world_size),
            **({"disable_lora": True} if disable_lora else {}),
        )
        return response_token_scores(frame, features)

    def _privileged_feature(self, row: DistillRow) -> dict:
        """Encode this row's privileged teacher view (``teacher_prompt`` + the shared response tokens)."""
        teacher_prompt = (row.extra or {}).get("teacher_prompt")
        if not teacher_prompt:
            raise ValueError(
                "OPSD requires a per-sample `teacher_prompt` carrying the privileged context; "
                "a row in this batch has none."
            )
        return encode_privileged_feature(
            self.template, row.messages, teacher_prompt, response_ids_from_feature(row.feature), row.extra
        )

    @remote_function(dispatch="all", collect="none", lazy_collect=False, stash=("teacher_logits", "teacher_logps"))
    def forward_kwargs(self, rows: List[DistillRow], use_student: bool) -> Dict[str, Any]:
        """One update's kwargs: student inputs plus the teacher signal in the ``signal`` form.

        ``use_student`` is the update's on/off-policy flag -- only GKD reads it (``apply_sft_loss``
        adds the supervised CE term to a dataset-sourced update); the logps forms score the same rows
        either way.
        """
        from swift.dev.utils.scoring import copy_inputs

        features = [row.feature for row in rows]
        if self.signal == "logits":
            teacher = self.teachers[0]
            if self.offload:
                teacher.reload_to_gpu()
            try:
                logits = teacher.forward_only(
                    inputs=copy_inputs(features),
                    return_logits=True,
                    micro_batch_size=max(1, self.train_batch_size // teacher.device_mesh.data_world_size),
                    **({"disable_lora": True} if self.disable_lora else {}),
                )["logits"]
            finally:
                if self.offload:
                    teacher.offload_to_cpu()
            signal_kwargs = {
                "teacher_logits": logits,
                "topk": self.gkd_logits_topk,
                "apply_sft_loss": bool(self.sft_alpha > 0 and not use_student),
            }
        elif self.signal == "logps":
            privileged = [self._privileged_feature(row) for row in rows]
            signal_kwargs = {
                "teacher_logps": self._response_logps(self.teachers[0], privileged, disable_lora=self.disable_lora),
            }
        else:
            # A dict keeps the teacher axis separate from DP/micro-batch slicing of each sample channel.
            tasks = [
                (self.placements[index], lambda index=index: self._response_logps(self.teachers[index], features))
                for index in range(len(self.teachers))
            ]
            signal_kwargs = {"teacher_logps": dict(enumerate(_run_grouped(tasks)))}
            if self.teacher_weights is not None:
                # Mixture weights are global scalars; tuples are broadcast by DP dispatch.
                signal_kwargs["teacher_weights"] = tuple(self.teacher_weights)
        return {
            "inputs": copy_inputs(features),
            "gradient_accumulation_steps": 1,
            **signal_kwargs,
        }
