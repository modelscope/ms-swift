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

    Every owner also exposes a ``placement_key``: the identity of the DeviceGroup it forwards on, used by
    :func:`~swift.dev.recipe._scoring.overlap_by_placement` to tell which scorers may run concurrently.
    ``None`` is the sentinel for "the policy actor itself" (the raw student, or a wrapper that reuses it),
    so all policy-actor scorers share one key and serialize; a separate frozen model carries its own group
    name, so scorers on disjoint groups overlap. The raw student model has no wrapper, so a loop resolves
    its key with ``getattr(owner, 'placement_key', None)``.
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
        #: Forwards on the policy actor (adapter disabled), so it shares the policy's placement key.
        self.placement_key = None

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
        #: Forwards on the policy actor (its current weights), so it shares the policy's placement key.
        self.placement_key = None

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        return self.student.forward_only(inputs=inputs, **kwargs)


class FrozenModelTeacher:
    """A separate frozen model -- a local copy or a Ray actor on its own DeviceGroup -- as the teacher.

    ``offload=True`` keeps the teacher on CPU between forwards and reloads it only to score, the memory hook
    for GKD's full-vocab logits. The reload/offload wrap lives here so a loop's scoring code is identical
    whether or not the teacher offloads. ``model`` is exposed so reference synchronization (which mutates a
    distinct mutable reference) can reach the underlying model.
    """

    def __init__(self, model: Any, *, offload: bool = False, placement_key: Any = None):
        self.model = model
        self.offload = offload
        #: The DeviceGroup this frozen model occupies -- its ``remote_group``. Two frozen teachers built
        #: into the SAME group (MOPD's packed K teachers) share one key and serialize; distinct groups
        #: (spread) get distinct keys and overlap. A model with no group name falls back to a per-instance
        #: token so it is never mistaken for the policy actor (``None``) or for another frozen model.
        self.placement_key = placement_key if placement_key is not None else ('frozen', id(model))
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
