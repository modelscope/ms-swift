"""Shared teacher / reference scoring abstractions for the on-policy RL loops.

GRPO (reference + RLSD/SDAR teacher) and the distillation family (GKD / OPSD / MOPD) all score with a
frozen owner via ``forward_only``. That owner used to be a union -- ``None`` | the ``'disable_lora'``
sentinel string | a frozen model | a list of frozen models -- with sentinel and ``isinstance`` checks
scattered across every loop (RL_PLAN W29). This module replaces that union with a few objects that all
expose ONE structural contract, ``forward_only(inputs, **kwargs)``, so a loop calls its scorer uniformly
and never branches on what KIND of teacher it is. The raw student model satisfies the same contract, so
"score on the student" needs no wrapper either.

What is deliberately NOT unified here is the teacher SIGNAL. GKD scores full-vocab ``teacher_logits``;
OPSD/MOPD score response-only ``teacher_logps`` (MOPD: K channels fused by weighted mixture); GRPO's
RLSD/SDAR scores response-only ``teacher_logps``. Those are three genuinely different contracts, and
folding them into one ``score()`` would change at least one loop's behaviour -- so each loop keeps its own
signal extraction and only the "who forwards, with the adapter disabled, with offload" resolution is
converged here.

The offline preference (``run_dpo``) and PPO loops still carry their own ``'disable_lora'`` sentinel; they
are independent loop classes whose convergence is a later stage (RL_PLAN §9 阶段三/四), not this module's.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Protocol

#: Dataset columns forwarded to ``template.encode`` when building a privileged teacher view, so a multimodal
#: sample's teacher forward sees the same non-text inputs as the student.
_MULTIMODAL_KEYS = ('images', 'videos', 'audios', 'tools', 'objects', 'chat_template_kwargs')


class Teacher(Protocol):
    """A scoring owner: anything that can ``forward_only`` a batch of features.

    Structural on purpose -- the student model itself already has ``forward_only``, so a loop may score on
    the raw student (dynamic self-distillation, or recomputing old log-probs) or on one of the wrappers
    below without ever branching on the type.
    """

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        ...


class DisableAdapterTeacher:
    """The student's OWN frozen base, scored by forwarding on the student actor with the adapter disabled.

    Replaces the ``'disable_lora'`` sentinel: a LoRA student distils from (or KL-anchors to) its
    adapter-disabled base, which needs no separate model or DeviceGroup -- just ``disable_lora=True`` on the
    student's own forward.
    """

    def __init__(self, student: Any):
        self.student = student

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        return self.student.forward_only(inputs=inputs, disable_lora=True, **kwargs)


class DynamicSelfTeacher:
    """The student's CURRENT (mutating) weights as the teacher -- on-policy self-distillation.

    Behaviourally a plain forward on the student, but a distinct type so OPSD's "no separate teacher" case
    is explicit rather than an overloaded ``None`` (which means "no teacher at all" in GRPO, where scoring
    is skipped unless a privileged ``teacher_prompt`` column is present).
    """

    def __init__(self, student: Any):
        self.student = student

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        return self.student.forward_only(inputs=inputs, **kwargs)


class FrozenModelTeacher:
    """A separate frozen model -- a local copy or a Ray actor on its own DeviceGroup -- as the teacher.

    ``offload=True`` keeps the teacher on CPU between forwards and reloads it only to score, the memory hook
    for GKD's full-vocab logits. The reload/offload wrap lives here so a loop's scoring code is identical
    whether or not the teacher offloads. ``model`` is exposed so reference synchronization (which mutates a
    distinct mutable reference) can reach the underlying model.
    """

    def __init__(self, model: Any, *, offload: bool = False):
        self.model = model
        self.offload = offload
        if offload:
            model.offload_to_cpu()

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        if not self.offload:
            return self.model.forward_only(inputs=inputs, **kwargs)
        self.model.reload_to_gpu()
        try:
            return self.model.forward_only(inputs=inputs, **kwargs)
        finally:
            self.model.offload_to_cpu()


class MultiTeacher:
    """K frozen domain teachers (MOPD) plus their mixture weights.

    Not a :class:`Teacher` itself -- it does not forward a single batch -- but a container the MOPD loop
    iterates, scoring each member teacher over the student's own features to produce one ``teacher_logps``
    channel each, which ``MOPDLoss`` fuses by ``weights``.
    """

    def __init__(self, teachers: List[Teacher], weights: Optional[List[float]] = None):
        self.teachers = list(teachers)
        self.weights = weights


def response_positions(feature: Dict[str, Any]) -> List[int]:
    """Indices of the trainable (response) tokens in ``feature``, i.e. where ``labels != -100``.

    This is the exact frame the loss mask uses, so teacher log-probs extracted at these positions align
    one-to-one with the student's per-token loss.
    """
    labels = feature.get('labels')
    if labels is None:
        return []
    return [idx for idx, label in enumerate(labels) if int(label) != -100]


def encode_privileged_view(
    template: Any,
    messages: List[dict],
    response_ids: List[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Encode a privileged teacher view (``messages`` carrying the shared ``response_ids``) and assert alignment.

    Shared by GRPO's RLSD/SDAR teacher view and OPSD's, which differ only in HOW ``messages`` is built
    (multi-turn reconstruction vs. a single privileged user turn) but not in the encode-and-check tail.
    Encoded in ``train`` mode with ``add_eos=False`` so the assistant turn is exactly ``response_ids`` (no
    extra end-of-sequence token), then checked: the encoded feature must expose the SAME number of trainable
    response positions as ``response_ids``. If a template or tokenizer difference changes that count, teacher
    and student would no longer score identical tokens and the alignment would be silently wrong -- so it is
    a hard error, not a truncation.
    """
    if template is None:
        raise ValueError('teacher_prompt requires a template to build the privileged teacher view.')
    extra = extra or {}
    row: Dict[str, Any] = {'messages': messages, 'add_eos': False}
    for key in _MULTIMODAL_KEYS:
        if extra.get(key) is not None:
            row[key] = extra[key]
    teacher_template = copy.copy(template)
    teacher_template.set_mode('train')
    feature = teacher_template.encode(row)
    if len(response_positions(feature)) != len(response_ids):
        raise RuntimeError('teacher_prompt encoding changed the response-token count; teacher and student must '
                           'share a tokenizer and score exactly the same response tokens.')
    return feature
