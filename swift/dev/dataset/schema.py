# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import numbers
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Tuple

from .format_converter.base import FormatConverter

__all__ = ['TASK_KEY_SPEC', 'TaskKeySpec', 'resolve_objective', 'validate_standard_row']

# Modes that run the model rather than train it: the row is only a prompt, so just `messages` is
# required regardless of task_type. Mirrors Template.is_training (swift/template/base.py).
_INFERENCE_MODES = frozenset({'transformers', 'vllm', 'lmdeploy', 'sglang'})

# The role vocabulary is FormatConverter.STANDARD_ROLES -- re-declaring it here would be a second
# source of truth free to drift from the converters that produce the messages.
_STANDARD_ROLES = frozenset(FormatConverter.STANDARD_ROLES)


def _is_message(value: Any) -> bool:
    return isinstance(value, dict) and 'role' in value and 'content' in value


def _is_messages(value: Any) -> bool:
    """One conversation: a non-empty list of ``{role, content}`` messages."""
    return isinstance(value, list) and len(value) > 0 and all(_is_message(m) for m in value)


def _is_message_groups(value: Any) -> bool:
    """A list of conversations (``positive_messages`` / ``negative_messages``); may itself be empty.

    Outer non-emptiness is the required / any-of checks' job, so an optional
    ``negative_messages=[]`` still passes its type check.
    """
    return isinstance(value, list) and all(_is_messages(group) for group in value)


def _is_bool_like(value: Any) -> bool:
    """A KTO good/bad label: ``bool`` or int (incl. numpy integers) -- the encoder coerces with
    ``bool()``. A str/float is rejected rather than silently coerced (``bool('false')`` is True)."""
    return isinstance(value, numbers.Integral)


@dataclass(frozen=True)
class TaskKeySpec:
    """The row contract for one objective.
    ``required``: present AND non-empty (emptiness is itself the error, e.g. ``messages``);
    ``present``: present but may be falsy (``0``/``False``/``[]`` are real label values);
    ``any_of``: at least one member present and non-empty (either-spelling keys);
    ``types``: key -> (predicate, expected-phrase), checked only when the key is present;
    ``doc``: one line echoed into every error raised for this objective.
    """
    required: Tuple[str, ...] = ()
    present: Tuple[str, ...] = ()
    any_of: Tuple[str, ...] = ()
    types: Dict[str, Tuple[Callable[[Any], bool], str]] = field(default_factory=dict)
    doc: str = ''


# Keyed by OBJECTIVE, not raw task_type: one task_type means different things in different modes
# (`causal_lm` is SFT in 'train' but a preference pair in 'rlhf'), so resolve_objective folds first.
TASK_KEY_SPEC: Dict[str, TaskKeySpec] = {
    'infer':
    TaskKeySpec(required=('messages', ), doc='Inference / rollout: a prompt, so only the dialogue is required.'),
    'causal_lm':
    TaskKeySpec(required=('messages', ), doc='SFT / PT (and prm): the dialogue is the whole contract.'),
    'rlhf':
    TaskKeySpec(
        required=('messages', ),
        any_of=('rejected_messages', 'rejected_response'),
        # No type check on the optional reward-modelling 'margin': the encoder coerces with
        # float(), which accepts more than any predicate; float() crashes loudly on bad types.
        doc='DPO/CPO/ORPO/SimPO and reward modelling: the chosen dialogue plus its rejected counterpart, '
        "as full turns ('rejected_messages') or a single reply ('rejected_response')."),
    'kto':
    TaskKeySpec(
        required=('messages', ),
        present=('label', ),
        types={'label': (_is_bool_like, 'a bool (or 0/1 int)')},
        doc='KTO: the dialogue plus a boolean good/bad label; the rejected side is optional.'),
    'seq_cls':
    TaskKeySpec(
        required=('messages', ),
        present=('label', ),
        # No type check: the label shape is problem_type-dependent (scalar or list) and the encoder
        # casts and fails loudly itself -- a predicate here could only false-reject.
        doc='Sequence classification / regression: the dialogue plus a label (a scalar, or a list for '
        'multi-label / multi-regression).'),
    'embedding':
    TaskKeySpec(
        required=('messages', 'positive_messages'),
        types={
            'positive_messages': (_is_message_groups, 'a list of conversations'),
            'negative_messages': (_is_message_groups, 'a list of conversations')
        },
        doc='Embedding: an anchor plus at least one positive; hard negatives are optional.'),
    'reranker':
    TaskKeySpec(
        required=('messages', ),
        any_of=('positive_messages', 'negative_messages'),
        types={
            'positive_messages': (_is_message_groups, 'a list of conversations'),
            'negative_messages': (_is_message_groups, 'a list of conversations')
        },
        doc='Reranker: a query plus at least one of positives / negatives to score against it.'),
    # jev's `decision` task is legacy-only for now; when ported it needs no entry here: its
    # messages-only contract (task metadata rides in extra_kwargs, unvalidated) is causal_lm's.
}


def resolve_objective(task_type: Optional[str], mode: Optional[str]) -> str:
    """Fold ``(task_type, mode)`` into the objective name that keys :data:`TASK_KEY_SPEC`.

    Mirrors ``Template.encode``'s dispatch: an inference mode needs only a prompt for every task
    (the label/positive consumption is gated on ``is_training``); embedding/reranker by task_type
    alone; the RL objectives by mode ('rlhf'/'kto'), which re-interprets ``causal_lm`` (DPO) and
    ``seq_cls`` (reward modelling). The rest falls back to messages-only.
    """
    if mode in _INFERENCE_MODES:
        return 'infer'
    if task_type == 'embedding':
        return 'embedding'
    if task_type in ('reranker', 'generative_reranker'):
        return 'reranker'
    if mode == 'rlhf':
        return 'rlhf'
    if mode == 'kto':
        return 'kto'
    if task_type == 'seq_cls':
        return 'seq_cls'
    return 'causal_lm'


def validate_standard_row(row: Dict[str, Any], task_type: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Raise ``ValueError`` unless ``row`` is a valid v5 standard row for ``(task_type, mode)``.

    Read-only and open to ``extra_kwargs``: only ``messages`` and the keys this task's
    :class:`TaskKeySpec` declares are inspected; an undeclared row-level key passes through
    unjudged. Called from the dev template's ``encode`` seam, the funnel every task's rows cross.
    """
    task_type = task_type or 'causal_lm'
    objective = resolve_objective(task_type, mode)
    spec = TASK_KEY_SPEC[objective]

    messages = row.get('messages')
    if not _is_messages(messages):
        _fail(task_type, objective, spec, f"'messages' must be a non-empty list of {{role, content}} dicts, "
              f'got {_brief(messages)}.')
    for message in messages:
        if message.get('role') not in _STANDARD_ROLES:
            _fail(task_type, objective, spec, f"message role {message.get('role')!r} is not one of "
                  f'{sorted(_STANDARD_ROLES)}: {_brief(message)}.')
        if message.get('content') is None:
            _fail(task_type, objective, spec, f"a message's 'content' is None: {_brief(message)}.")

    # Inline media and the top-level media columns are mutually exclusive; checked readably here
    # instead of mid-encode (forward-moved from StdTemplateInputs.from_dict's bare assert).
    for kind in sorted(_inline_media_kinds(messages)):
        if row.get(kind):
            _fail(task_type, objective, spec, f"the row carries media BOTH inline in 'messages' and in the "
                  f"top-level '{kind}' column. Use one form only: inline content parts (canonical) or "
                  f"'<{kind[:-1]}>' placeholders plus the '{kind}' column.")

    for key in spec.required:
        if _is_absent(row.get(key)):
            _fail(task_type, objective, spec, f"missing required key '{key}'.")
    for key in spec.present:
        # Presence-only: a falsy value (0 / False / []) is a valid label, not a missing one.
        if key not in row or row[key] is None:
            _fail(task_type, objective, spec, f"missing required key '{key}'.")
    if spec.any_of and all(_is_absent(row.get(key)) for key in spec.any_of):
        options = ' or '.join(repr(key) for key in spec.any_of)
        _fail(task_type, objective, spec, f'missing required key: at least one of {options} must be set.')

    for key, (predicate, expected) in spec.types.items():
        value = row.get(key)
        if value is not None and not predicate(value):
            _fail(task_type, objective, spec, f"key '{key}' should be {expected}, got {_brief(value)}.")


def _is_absent(value: Any) -> bool:
    """Absent for a ``required`` key: missing/None, or an empty list/str/dict/tuple. Falsy-but-valid
    scalars (``0``/``False``/``[]`` as a label) are NOT judged here -- those go in ``present``."""
    if value is None:
        return True
    return isinstance(value, (list, str, dict, tuple)) and len(value) == 0


def _inline_media_kinds(messages: List[Dict[str, Any]]) -> Set[str]:
    """The top-level media column names that ``messages`` carries INLINE, as content parts.

    Mirrors ``StdTemplateInputs.remove_messages_media`` so the mixing check agrees with the
    encoder's assert: str / pre-encoded content holds no media; a non-'text' part resolves its
    value (unwrapping ``{'url': ...}``) and counts toward a kind ONLY when truthy -- the encoder's
    ``if value:`` gate. Counting by part TYPE alone would false-report mixing.
    """
    kinds: Set[str] = set()
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list) or not content or isinstance(content[0], int):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get('type')
            if not part_type or part_type == 'text':
                continue
            value = part.get(part_type)
            if isinstance(value, dict):
                value = value.get('url')
            if not value:
                continue
            kind = part_type[:-len('_url')] if part_type.endswith('_url') else part_type
            kinds.add(f'{kind}s')
    return kinds


def _brief(value: Any, limit: int = 200) -> str:
    """A short, safe repr for error messages -- a full multimodal row would otherwise flood the log."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + '...'


def _fail(task_type: str, objective: str, spec: TaskKeySpec, detail: str) -> NoReturn:
    raise ValueError(f"[v5 schema] task_type='{task_type}' (objective='{objective}'): {detail}\n"
                     f'  This task expects: {spec.doc}\n'
                     f"  See the 'v5 standard dataset format' docs for the full row schema.")
