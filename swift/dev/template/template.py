# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, List, Optional

from swift.template.base import Template as LegacyTemplate
from swift.dev.utils import get_logger

logger = get_logger()


class DevMixin:
    """dev's entire delta over a legacy Template: the twinkle contract, nothing else.

    Everything the encoding itself does stays in swift's legacy Template -- this mixin only adds what
    twinkle's Model requires and the label convention twinkle's forward assumes. It is DEVELOPMENT-ONLY
    scaffolding: once these two behaviours move into swift's Template proper, the mixin (and this
    module) disappear.

    Mixed into the REAL legacy class by `shifted_template_class`, so every method the family overrode
    (`_encode`, `replace_tag`, `_data_collator`, `_get_position_ids`, `packing_row`, ...) keeps
    dispatching. Must precede the legacy class in the MRO.

    Two additions:

    1. Label convention (`encode`). swift's legacy Template emits labels ALIGNED with input_ids
       (labels[i] == input_ids[i]) and shifts at loss time (HF convention). twinkle's forward computes
       logps via no-shift selective_log_softmax and therefore expects NEXT-token labels, which
       twinkle's own Template produces at encode time (_roll_labels). To stay consistent with twinkle
       ("whoever encodes, shifts") the shift happens here -- the InputProcessor must NOT touch label
       semantics.

    2. `batch_encode`, the only twinkle Template method a swift Template lacks. twinkle's Model calls
       it to turn raw trajectories into features (transformers.py / megatron.py, guarded by
       `_not_encoded`); `processor` and `pre_forward_hook`, the rest of the contract, already exist on
       swift's Template. Without it the model cannot use a swift template at all
       (AttributeError on the first step), which is why the model used to fall back to its own
       default twinkle Template and silently ignore every TemplateConfig field.
    """

    # Records that a sample is already shifted, so a second encode cannot shift twice.
    SHIFTED_KEY = '_labels_shifted'

    @staticmethod
    def _shift_labels_next_token(labels: List[int]) -> List[int]:
        """Explicit next-token shift: out[i]=labels[i+1], out[-1]=-100 (NOT circular roll)."""
        if not labels:
            return labels
        return list(labels[1:]) + [-100]

    # Tasks whose `labels` are NOT per-token targets, so the next-token shift must not fire.
    # embedding: `_embedding_encode` emits one label PER SEQUENCE (1.0 marks the anchor that starts
    #   an anchor/positive/negatives group, 0.0 the rest). Shifting would move the 1.0 marker off the
    #   front and append -100, and InfonceLoss locates groups via torch.nonzero(labels) -- so groups
    #   would split at the wrong offsets and train on silently wrong pairs.
    # reranker/seq_cls: likewise per-sequence scores/classes rather than token targets.
    _NO_SHIFT_TASK_TYPES = frozenset({'embedding', 'reranker', 'generative_reranker', 'seq_cls'})

    def encode(self, inputs, return_template_inputs: bool = False, return_length: bool = False):
        """Encode, then shift labels to next-token alignment (training mode only).

        vLLM-mode guard: the next-token shift must fire ONLY in training modes. In
        inference/rollout modes (vllm/lmdeploy/sglang/transformers) legacy `is_training` is False and
        `_encode` clears labels to None, so today the `labels is not None` check already skips the
        shift. We ALSO gate on `self.is_training` explicitly so a rollout input can never be silently
        shifted even if a future mode were to emit labels.

        Task guard: only `causal_lm` labels are per-token. See ``_NO_SHIFT_TASK_TYPES``.
        """
        encoded = super().encode(inputs, return_template_inputs=return_template_inputs, return_length=return_length)
        if (self.is_training and getattr(self, 'task_type', 'causal_lm') not in self._NO_SHIFT_TASK_TYPES
                and isinstance(encoded, dict) and encoded.get('labels') is not None
                and not encoded.get(self.SHIFTED_KEY)):
            encoded['labels'] = self._shift_labels_next_token(list(encoded['labels']))
            if encoded.get('loss_scale') is not None:
                encoded['loss_scale'] = list(encoded['loss_scale'][1:]) + [0.0]
            encoded[self.SHIFTED_KEY] = True
        return encoded

    def batch_encode(self, trajectories, add_generation_prompt: bool = False, **kwargs):
        """twinkle's batch entry point, delegated row-by-row to swift's `encode`.

        Deliberately thin: the point of routing the model through a swift template is that ONE encode
        implementation produces the training tokens, so this must not re-implement any of it.
        `add_generation_prompt` is a twinkle inference concept with no training-time counterpart in
        swift's encode (which derives it from the template mode); it is accepted so the signature
        matches twinkle's and rejected when set, rather than silently ignored.

        twinkle also accepts a columnar dict; only the row-list form is supported here because that is
        what the Model passes on the SFT path (it wraps a single dict into a list before calling).
        """
        if add_generation_prompt:
            raise NotImplementedError(
                'batch_encode(add_generation_prompt=True) is not supported on a swift template: the '
                'generation prompt is decided by the template mode (set_mode), not per call.')
        if isinstance(trajectories, dict):
            raise NotImplementedError('batch_encode expects a list of trajectories, not a columnar dict.')
        return [self.encode(dict(trajectory), **kwargs) for trajectory in trajectories]


# Cache keyed by legacy class: one derived class per family, so `isinstance` stays meaningful across
# templates and the class is created once.
_SHIFTED_CLASSES: Dict[type, type] = {}

# Reverse index (derived class NAME -> legacy base), used by this module's __getattr__ to rebuild a
# class when pickle looks it up by name in a process that never created it.
_SHIFTED_BASES: Dict[str, type] = {}


def shifted_template_class(base: type) -> type:
    """`base` + the next-token shift, as a class -- the default path's alternative to re-classing.

    Preserves the legacy class instead of replacing it. That distinction is the whole point:
    `Template.from_template` overwrites `__class__`, which drops every method the legacy subclass
    overrode (measured on Qwen3.5: 14, including `_encode`, `replace_tag`, `_data_collator`,
    `_get_position_ids`, `packing_row`) and silently routes `super()._encode()` to the BASE legacy
    `_encode` rather than the family's. Deriving keeps all of them and adds only `encode`.

    The class is registered in this module's globals under its own name so it stays PICKLABLE:
    `build_dataset` hands the template to EncodePreprocessor/AddLengthPreprocessor/PackingDataset,
    which datasets.map pickles whenever num_proc > 1, and pickle resolves classes by
    module + qualname. A plain `type(...)` result is unreachable that way and would fail there only.

    Registering in globals() is not sufficient on its own: it only populates the process that built
    the class, while datasets.map's workers are FRESH interpreters that merely import this module.
    Unpickling there looked the name up in a module where nothing had created it yet and raised
    `AttributeError: Can't get attribute 'ShiftedTemplate'`. The module-level `__getattr__` below
    closes that gap by rebuilding the class on demand, so `_SHIFTED_BASES` must stay in sync.
    """
    cls = _SHIFTED_CLASSES.get(base)
    if cls is None:
        name = f'Shifted{base.__name__}'
        cls = type(name, (DevMixin, base), {'__module__': __name__, '__qualname__': name})
        globals()[name] = cls
        _SHIFTED_CLASSES[base] = cls
        # Needed by __getattr__ to rebuild this exact class in a worker process, where the name is
        # all pickle has to go on.
        _SHIFTED_BASES[name] = base
    return cls


def __getattr__(name: str) -> type:
    """Rebuild a `Shifted<Family>` class on first lookup, so unpickling works in a fresh process.

    pickle stores a dynamically created class as module + qualname and resolves it with getattr on the
    imported module. In the process that called shifted_template_class the name is already in
    globals(); in a datasets.map worker (a new interpreter) it is not, and without this hook the load
    fails with AttributeError -- only under num_proc > 1, which is why it survived the single-process
    tests.

    The base class is recovered from legacy's template registry, since the name is all pickle carries:
    'Shifted' + the legacy class's __name__. That resolves to one class only because __name__ is
    injective over the registry -- measured at 229 entries / 116 distinct names, where every shared
    name is the SAME class serving several template types. The invariant is not enforced anywhere: if
    two different legacy classes ever share a __name__, this would rebuild from whichever the scan hits
    first and the worker would silently encode with the wrong family. A name that matches nothing
    raises AttributeError, as module attribute lookup must.
    """
    if not name.startswith('Shifted'):
        raise AttributeError(name)
    base = _SHIFTED_BASES.get(name)
    if base is None:
        base = _find_legacy_template_class(name[len('Shifted'):])
    if base is None:
        raise AttributeError(name)
    return shifted_template_class(base)


def _find_legacy_template_class(base_name: str) -> Optional[type]:
    """Locate a legacy Template subclass by its __name__, for __getattr__'s rebuild path.

    Checks the base class first (the common case -- most families do not subclass Template) and then
    walks legacy's TEMPLATE_MAPPING, which is the only enumeration of the concrete classes.
    """
    if LegacyTemplate.__name__ == base_name:
        return LegacyTemplate
    try:
        from swift.template.register import TEMPLATE_MAPPING
    except ImportError:
        return None
    for meta in TEMPLATE_MAPPING.values():
        cls = getattr(meta, 'template_cls', None)
        if cls is not None and cls.__name__ == base_name:
            return cls
    return None
