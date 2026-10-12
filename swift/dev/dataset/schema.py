# Copyright (c) ModelScope Contributors. All rights reserved.
"""The v5 standard-row schema: the single source of truth for what each task's rows must carry.

A *standard row* is the shape every dataset is normalised to before it reaches the template encoder:
``messages`` at the core (OpenAI-aligned; multimodal via inline content parts) plus the extra keys a
task appends on top. This module declares, per task, which of those keys are required, and checks a
row against them so a missing key fails loudly HERE -- naming the task, the key and the fix -- rather
than deep inside the encoder, where the same mistake surfaces either as a cryptic crash
(``inputs.rejected is None``) or, worse, as silently wrong training data (a ``seq_cls`` row with no
``label`` encodes to no targets at all).

Two properties are load-bearing and must not regress:

* **The required keys mirror what the encoder actually consumes.** They are read off the
  ``Template.encode`` dispatch in ``swift/template/base.py``: ``(task_type, mode)`` selects the
  objective, and each objective's ``_*_encode`` reads a fixed set of keys. This table is a
  *description* of that behaviour, not a new policy, so a row the encoder accepts always validates.

* **The schema is open to ``extra_kwargs``.** A row-level key this module does not declare is left
  untouched and unjudged -- it is the extension surface future tasks ride in on. ``jev``'s
  ``decision`` task, for example, carries ``span`` / ``verbalizer_ids`` / ``target_probs`` / ``kind``
  alongside its ``messages``; none are declared here and none may cause a row to be rejected. An
  unknown key is therefore NEVER an error: validation is a per-key check against what a task needs,
  not a closed-set rejection of the whole row.
"""
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

# A standard row carries media EITHER inline (content parts, the canonical v5 form) OR in these
# top-level columns (the v4 / built-in-loader compatible form), never both. Mirrors the keys
# StdTemplateInputs.remove_messages_media normalises inline media into.
_MEDIA_COLUMNS = ('images', 'videos', 'audios')


# -- type predicates (light callables; the schema deliberately pulls in no pydantic) --------------

def _is_message(value: Any) -> bool:
    return isinstance(value, dict) and 'role' in value and 'content' in value


def _is_messages(value: Any) -> bool:
    """One conversation: a non-empty list of ``{role, content}`` messages."""
    return isinstance(value, list) and len(value) > 0 and all(_is_message(m) for m in value)


def _is_message_groups(value: Any) -> bool:
    """A list of conversations (``positive_messages`` / ``negative_messages``); may itself be empty.

    Non-emptiness of the OUTER list is enforced by the required / any-of checks, not here, so an
    optional ``negative_messages=[]`` still passes its type check.
    """
    return isinstance(value, list) and all(_is_messages(group) for group in value)


def _is_bool_like(value: Any) -> bool:
    """A KTO good/bad label: a ``bool`` or an int. Python's ``bool`` is an ``int`` subclass and a
    loaded label may be a numpy integer, so ``numbers.Integral`` covers both; the encoder coerces with
    ``bool()``, so any integral value is accepted. A str/float is rejected here rather than silently
    coerced (``bool('false')`` is ``True``).
    """
    return isinstance(value, numbers.Integral)


# -- the per-objective contract -------------------------------------------------------------------

@dataclass(frozen=True)
class TaskKeySpec:
    """The row contract for one objective.

    Attributes:
        required: Keys that must be present AND non-empty -- for keys whose emptiness is itself an
            error (``messages``; an empty ``positive_messages`` would index ``[0]`` and crash the
            encoder). A falsy-but-valid scalar must NOT go here (see ``present``).
        present: Keys that must be present but may legitimately be falsy or empty -- a label whose
            ``0``, ``False`` or ``[]`` (a valid empty multi-label target) is a real value, not a
            missing one. Only absence / ``None`` counts as missing.
        any_of: A group of keys of which at least one must be present and non-empty. Used where the
            encoder accepts either of two spellings (``rejected_messages`` OR ``rejected_response``).
        types: ``key -> (predicate, expected-phrase)``, checked only when the key is present, so an
            optional key of the wrong type still fails loudly while an absent one is skipped.
        doc: One line describing the contract, echoed into every error raised for this objective.
    """
    required: Tuple[str, ...] = ()
    present: Tuple[str, ...] = ()
    any_of: Tuple[str, ...] = ()
    types: Dict[str, Tuple[Callable[[Any], bool], str]] = field(default_factory=dict)
    doc: str = ''


# Keyed by OBJECTIVE, not by raw task_type: one task_type means different things in different modes
# (`causal_lm` is plain SFT in 'train' but a preference pair in 'rlhf'), so `resolve_objective` folds
# (task_type, mode) into one of these names before the table is consulted.
TASK_KEY_SPEC: Dict[str, TaskKeySpec] = {
    'infer':
    TaskKeySpec(required=('messages', ), doc='Inference / rollout: a prompt, so only the dialogue is required.'),
    'causal_lm':
    TaskKeySpec(required=('messages', ), doc='SFT / PT (and prm): the dialogue is the whole contract.'),
    'rlhf':
    TaskKeySpec(
        required=('messages', ),
        any_of=('rejected_messages', 'rejected_response'),
        # No type check on the optional reward-modelling 'margin': the encoder coerces it with
        # float(), which accepts more than a numeric predicate would (e.g. a numeric string), so a
        # check here could only false-reject; float() itself crashes loudly on a genuinely bad type.
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
        # No type check: the label shape is problem_type-dependent -- a scalar (single-label /
        # single-regression) or a list (multi-label / multi-regression, possibly empty) -- and the
        # validator does not know problem_type. The encoder casts it (int() for single-label) and
        # fails loudly on a genuinely bad type, so a predicate here could only false-reject.
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
    # Future-compat entry for jev's `decision` task (still legacy-only). It is deliberately just
    # `messages`: its task-specific metadata rides in extra_kwargs and is neither declared nor
    # validated here, so porting jev into dev needs no schema change.
    'decision':
    TaskKeySpec(
        required=('messages', ),
        doc='jev decision (future-compat): the dialogue only; task-specific metadata '
        '(span/verbalizer_ids/target_probs/kind) rides in extra_kwargs, undeclared and unvalidated.'),
}


def resolve_objective(task_type: Optional[str], mode: Optional[str]) -> str:
    """Fold ``(task_type, mode)`` into the objective name that keys :data:`TASK_KEY_SPEC`.

    Mirrors the dispatch in ``Template.encode``, whose OUTER branch is ``task_type`` and whose ``mode``
    branch lives only inside ``causal_lm`` and ``seq_cls``. So: an inference mode needs just a prompt
    for every task; ``embedding``/``reranker``/``decision`` are chosen by task_type alone (``mode``
    never re-interprets a contrastive run); the RL objectives are then chosen by MODE ('rlhf'/'kto'),
    which is what the encoder branches on for ``causal_lm`` (DPO family) and ``seq_cls`` (reward
    modelling) -- the same ``causal_lm`` is SFT in 'train' but a preference pair in 'rlhf'. Anything
    else (``causal_lm``, ``prm``, an unknown task_type) falls back to the messages-only contract, so a
    future task is never rejected for keys this table has not heard of.
    """
    if mode in _INFERENCE_MODES:
        return 'infer'
    if task_type == 'embedding':
        return 'embedding'
    if task_type in ('reranker', 'generative_reranker'):
        return 'reranker'
    if task_type == 'decision':
        return 'decision'
    if mode == 'rlhf':
        return 'rlhf'
    if mode == 'kto':
        return 'kto'
    if task_type == 'seq_cls':
        return 'seq_cls'
    return 'causal_lm'


# -- validation -----------------------------------------------------------------------------------

def validate_standard_row(row: Dict[str, Any], task_type: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Raise ``ValueError`` unless ``row`` is a valid v5 standard row for ``(task_type, mode)``.

    Read-only (never mutates ``row``) and open to ``extra_kwargs``: only ``messages`` and the keys this
    task's :class:`TaskKeySpec` declares are inspected, so an undeclared row-level key -- a future
    task's metadata, a reward function's reference column -- is passed through unjudged. Called from
    the dev template's ``encode`` seam, the single funnel every task's rows flow through.
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

    # Inline media and the top-level media columns are mutually exclusive; forward-moved from
    # StdTemplateInputs.from_dict's bare assert so it fails readably, here, instead of mid-encode.
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
    """Whether a ``required`` key is absent: missing/None, or an empty list/str/dict/tuple.

    Used only for keys whose emptiness is itself an error (``messages``, ``positive_messages``). A
    falsy-but-valid scalar (``0`` / ``False`` / ``[]`` as a label) is NOT judged here -- those go in
    ``TaskKeySpec.present``, which tests presence only.
    """
    if value is None:
        return True
    return isinstance(value, (list, str, dict, tuple)) and len(value) == 0


def _inline_media_kinds(messages: List[Dict[str, Any]]) -> Set[str]:
    """The top-level media column names that ``messages`` carries INLINE, as content parts.

    Mirrors ``StdTemplateInputs.remove_messages_media`` so the mixing check agrees with the encoder's
    own ``assert``: a str content and a pre-encoded one (``list[int]``, or a dict with ``token_ids``)
    hold no inline media; otherwise each non-'text' part resolves its value (``part[part['type']]``,
    unwrapping ``{'url': ...}``) and counts toward a kind ONLY when that value is truthy -- exactly the
    encoder's ``if value: res[f'{key}s'].append(value)`` gate. Counting by part TYPE alone would treat
    an ``image_url`` part with an empty url as inline media and falsely report mixing against a
    top-level ``images`` column the encoder would in fact have used.
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
