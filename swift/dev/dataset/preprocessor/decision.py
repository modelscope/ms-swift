# Copyright (c) ModelScope Contributors. All rights reserved.
"""Preprocessor base for the `decision` task_type (typed-decision System-1 scoring models).

A typed-decision record carries a `state` plus one or more questions; each question has a
`kind in {noul, choice, score}` and its own option set. `ScoringPreprocessor` normalizes a raw
dataset row into swift's `messages` plus the decision fields, which reach the Template through
`StdTemplateInputs.extra_kwargs` (non-standard keys are auto-collected by `from_dict`, so no schema
change is needed). The model template `_encode` renders the actual prompt from these fields and
records the per-question token spans; `target_probs` becomes the training labels.

Output row contract (see also `swift/model/decision_head.py`):
    messages:     [{'role': 'user', 'content': state}]  a valid message list (from_dict requires one);
                  the template re-renders the full decision layout from the fields below.
    questions:    List[str]           per-question text
    kinds:        List[str]           per-question 'noul' | 'choice' | 'score'
    options:      List[List[str]]     per-question option texts (noul -> ['false','true'], score -> ['0'..'5'])
    target_probs: List[List[float]]   per-question gold distribution over its OWN options (training only)

Subclasses override `parse_record` to turn a model-specific raw row into `(state, questions)`,
where each question is a dict `{'kind', 'question', 'options', 'gold'}` and `gold` is either an
option index (int) or a full distribution (sequence of floats, for distillation targets).
"""

import random as _random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json

from swift.dataset.preprocessor.core import RowPreprocessor
from swift.dev.utils import get_logger

logger = get_logger()


def _startlux_norm(s: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


@dataclass
class DecisionAugmentConfig:
    """Training-time schema augmentation for decision datasets"""

    drop_instructions: float = 0.0
    subsample_options: int = 0
    state_as_text: float = 0.0
    shuffle_fields: bool = False


def _humanize(label: str) -> str:
    return re.sub(r"[_\-/]+", " ", str(label)).strip()


def _state_to_text(state: Any) -> str:
    if not isinstance(state, dict):
        if isinstance(state, str):
            return state
        return json.dumps(state, ensure_ascii=False)
    if len(state) == 0:
        return ""
    if len(state) == 1:
        return str(next(iter(state.values())))
    return "\n".join(f"{_humanize(key).capitalize()}: {value}" for key, value in state.items())


def _subsample_options(
    question: Dict[str, Any], gold: Any, rng: _random.Random, max_options: int
) -> Tuple[Dict[str, Any], Any]:
    """Subsample choice options to ``max_options``, always keeping the highest-probability option.

    Only applies to ``choice`` questions whose options exceed ``max_options``. For an int gold
    (one-hot), the gold index is always kept and ``max_options - 1`` distractors are sampled.
    For a distribution gold (soft labels), the argmax option is always kept and the distribution
    is truncated and renormalized over the kept options. ``noul`` / ``score`` questions have
    fixed option sets and are never subsampled.
    """
    options = question.get("options")
    if not options or len(options) <= max_options:
        return question, gold
    kind = question.get("kind")
    if kind in ("noul", "score"):
        return question, gold
    # OmniJev keeps its answer keys out-of-band (``column_keys``, one entry MORE than ``options``
    # for an abstain choice question) and gold indexes those, so subsampling ``options`` would
    # re-index the label against a key set that no longer matches the rendered options.
    if "column_keys" in question:
        return question, gold

    if isinstance(gold, int) and 0 <= gold < len(options):
        peak = gold
    elif isinstance(gold, (list, tuple)) and len(gold) == len(options):
        peak = max(range(len(gold)), key=lambda i: gold[i])
    else:
        return question, gold

    others = [i for i in range(len(options)) if i != peak]
    kept = set(rng.sample(others, min(max_options - 1, len(others)))) | {peak}
    kept = sorted(kept)
    new_options = [options[i] for i in kept]

    if isinstance(gold, int):
        new_gold: Any = kept.index(gold)
    else:
        new_gold = [float(gold[i]) for i in kept]
        total = sum(new_gold)
        if total > 0:
            new_gold = [p / total for p in new_gold]

    new_q = {**question, "options": new_options}
    return new_q, new_gold


def _augment_row_key(state: Any, questions: List[Dict[str, Any]]) -> str:
    """A stable fingerprint of one parsed record: each row's augment RNG is derived from it, so an
    identical record is augmented identically no matter which shard, worker or resume sees it."""
    return json.dumps([state, questions], ensure_ascii=False, sort_keys=True, default=str)


def augment_decision_row(  # noqa: C901
    state: Any,
    questions: List[Dict[str, Any]],
    config: Optional[DecisionAugmentConfig],
    rng: Optional[_random.Random] = None,
    state_field_order_fixed: bool = False,
) -> Tuple[Any, List[Dict[str, Any]]]:
    if config is None:
        return state, questions
    if (
        config.drop_instructions == 0.0
        and config.subsample_options <= 0
        and config.state_as_text == 0.0
        and not config.shuffle_fields
    ):
        return state, questions
    rng = rng or _random.Random(42)

    # drop_instructions: set the question text to empty (template will still render the layout,
    # model must infer intent from options only).
    if config.drop_instructions > 0:
        for q in questions:
            if rng.random() < config.drop_instructions:
                q["question"] = ""

    # subsample_options: keep gold + N distractors for choice questions.
    if config.subsample_options > 0:
        for i, q in enumerate(questions):
            new_q, new_gold = _subsample_options(q, q.get("gold"), rng, config.subsample_options)
            if new_gold is None:
                new_q.pop("gold", None)  # a question without gold stays keyless, not gold=None
            else:
                new_q["gold"] = new_gold
            questions[i] = new_q

    # state_as_text: randomly flatten dict state to plain text so the model sees different
    # renderings (JSON vs human-readable lines). Per-record probability.
    if config.state_as_text > 0 and isinstance(state, dict) and state:
        if rng.random() < config.state_as_text:
            state = _state_to_text(state)

    # shuffle_fields: randomize field order in dict state AND question order. The state-side
    # shuffle makes the model robust to permutation of schema fields ; the question-side shuffle
    # only matters for multi-question records
    if config.shuffle_fields:
        if not state_field_order_fixed and isinstance(state, dict) and len(state) > 1:
            keys = list(state.keys())
            rng.shuffle(keys)
            state = {k: state[k] for k in keys}
        if len(questions) > 1:
            rng.shuffle(questions)

    return state, questions


class ScoringPreprocessor(RowPreprocessor):
    # The augmentation channel this preprocessor consumes; the loader binds the run's
    # ``{'decision': DecisionAugmentConfig}`` entry via ``bind_augment`` (see Preprocessor).
    augment_channel = "decision"

    # Run-level augmentation state, bound together by ``bind_augment``; until then ``_augment``
    # is a no-op (a None config short-circuits before any RNG is built).
    augment_config: Optional[DecisionAugmentConfig] = None
    _augment_seed: Optional[int] = None

    # True -> fan a record out into ONE row per question (each question becomes its own prompt
    # holding the whole state); False -> one row per record carrying all its questions.
    fan_out = False

    # Name of the model family this preprocessor targets; prefixes its error messages. Every
    # concrete subclass sets it (the base class is abstract).
    MODEL_NAME: Optional[str] = None

    # Whether the row's state rendering FIXES a dict state's field order (key-sorted JSON). When
    # True, ``shuffle_fields`` skips the state side: the encoding would throw the shuffled order
    # away, so shuffling there is dead work.
    state_field_order_fixed = False

    NOUL_OPTIONS = ["false", "true"]
    SCORE_OPTIONS = ["0", "1", "2", "3", "4", "5"]

    # Canonical aliases for question kinds.
    KIND_ALIASES = {
        "noul": "noul",
        "yes_no": "noul",
        "yn": "noul",
        "bool": "noul",
        "choice": "choice",
        "multiple_choice": "choice",
        "mcq": "choice",
        "select": "choice",
        "score": "score",
        "rating": "score",
        "ordinal": "score",
        "likert": "score",
    }

    def __init__(self, **kwargs) -> None:
        kwargs.pop("seed", None)
        super().__init__(**kwargs)

    def bind_augment(self, config: DecisionAugmentConfig, seed: int) -> None:
        """Store the run-level augmentation config and the dataset seed backing the per-row RNGs"""
        self.augment_config = config
        self._augment_seed = int(seed)
        logger.info_once(
            f"Decision augmentation enabled on {type(self).__name__}: "
            f"drop_instructions={config.drop_instructions}, "
            f"subsample_options={config.subsample_options}, "
            f"state_as_text={config.state_as_text}, "
            f"shuffle_fields={config.shuffle_fields}, seed={seed}."
        )

    def _augment(self, state: Any, questions: List[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
        if self.augment_config is None:
            return state, questions
        # Per-row RNG seeded from (run seed, content fingerprint): identical records are augmented
        # identically across shards, dataloader workers and resume runs, and independently of
        # processing order. The fingerprint replaces a shared sequential RNG whose draws depended
        # on which rows happened to be visited first.
        row_rng = _random.Random(f"{self._augment_seed}:{_augment_row_key(state, questions)}")
        return augment_decision_row(
            state, questions, self.augment_config, row_rng, state_field_order_fixed=self.state_field_order_fixed
        )

    def preprocess(self, row: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        state, questions = self.parse_record(row)
        if not questions:
            return None
        state, questions = self._augment(state, questions)
        if self.fan_out:
            return [self._build_row(row, state, [q]) for q in questions]
        return self._build_row(row, state, questions)

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Split out of `preprocess` so a subclass can fan a record out into several rows (JEV renders one
        question per prompt) while reusing this row-building logic unchanged"""
        kinds = [q["kind"] for q in questions]
        options = [self._resolve_options(q) for q in questions]
        out: Dict[str, Any] = {
            "messages": [{"role": "user", "content": self._state_to_text(state)}],
            "questions": [q["question"] for q in questions],
            "kinds": kinds,
            "options": options,
        }
        for key in ("images", "videos", "audios"):
            if row.get(key) is not None:
                out[key] = row[key]
        if self._has_gold(questions):
            target_probs: List[List[float]] = []
            for q, opt in zip(questions, options):
                gold = q.get("gold")
                if gold is None:
                    # Partially-labeled record: keep the unlabeled question as an all-zero target
                    # row (the loss masks zero rows), instead of dropping the whole record.
                    target_probs.append([0.0] * len(opt))
                else:
                    target_probs.append(self.to_target_dist(gold, len(opt)))
            out["target_probs"] = target_probs
        return out

    @classmethod
    def _normalize_kind(cls, kind: Any) -> str:
        key = str(kind).strip().lower()
        if key not in cls.KIND_ALIASES:
            raise ValueError(f"unknown question kind {kind!r}; expected one of noul/choice/score (or an alias).")
        return cls.KIND_ALIASES[key]

    @staticmethod
    def _question_items(questions_raw: Any) -> List[Tuple[Any, Any]]:
        if isinstance(questions_raw, dict):
            return list(questions_raw.items())
        if isinstance(questions_raw, (list, tuple)):
            return list(enumerate(questions_raw))
        return []

    @staticmethod
    def _state_to_text(state: Any) -> str:
        """Coerce the ``state`` value to a plain string for the message ``content``"""
        if isinstance(state, str):
            return state
        if state is None:
            return ""
        if isinstance(state, dict):
            parts = [v for v in state.values() if isinstance(v, str) and v.strip()]
            return "\n".join(parts)
        return str(state)

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        """Return `(state, questions)`; each question is `{'kind','question','options','gold'}`.

        `options` may be omitted for noul/score (defaults are filled by `_resolve_options`); `gold`
        may be None at inference. Must be implemented by each model subclass.
        """
        raise NotImplementedError

    def _resolve_options(self, question: Dict[str, Any]) -> List[str]:
        options = question.get("options")
        if options:
            return list(options)
        default = self.default_options(question["kind"])
        if default is None:
            raise ValueError(f"A 'choice' question must provide its options: {question}")
        return default

    @classmethod
    def default_options(cls, kind: str) -> Optional[List[str]]:
        if kind == "noul":
            return list(cls.NOUL_OPTIONS)
        if kind == "score":
            return list(cls.SCORE_OPTIONS)
        return None

    @staticmethod
    def _has_gold(questions: Sequence[Dict[str, Any]]) -> bool:
        return any(q.get("gold") is not None for q in questions)

    @staticmethod
    def to_target_dist(gold: Any, n_options: int) -> List[float]:
        """Gold -> a distribution over the question's own `n_options` options.

        An int gold is the correct option index (one-hot); a sequence gold is already a full
        distribution (JEV distillation), and must match `n_options`.
        """
        if isinstance(gold, (list, tuple)):
            probs = [float(x) for x in gold]
            if len(probs) != n_options:
                raise ValueError(f"target distribution length {len(probs)} != n_options {n_options}")
            return probs
        if isinstance(gold, float) and not float(gold).is_integer():
            raise ValueError(
                f"gold {gold!r} is not an integral option index; a float gold must be "
                "a whole-number index or a full probability distribution"
            )
        idx = int(gold)
        if not 0 <= idx < n_options:
            raise ValueError(f"gold index {idx} out of range for n_options {n_options}")
        dist = [0.0] * n_options
        dist[idx] = 1.0
        return dist

    @staticmethod
    def _first(source: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
        """Return the first non-None value among `keys` (a permissive column lookup shared by the
        model subclasses, whose datasets use differing column names).
        """
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
        return default

    def _lookup_gold(self, question: Dict[str, Any], answers: Any, qid: Any) -> Any:
        gold = self._first(question, self.GOLD_KEYS, default=None)
        if gold is None and isinstance(answers, dict):
            gold = answers.get(str(qid), answers.get(qid))
        return gold

    @staticmethod
    def _as_option_dict(option: Any) -> Dict[str, Any]:
        """Normalize ONE unified option element to a dict."""
        if isinstance(option, str):
            return {"text": option}
        if isinstance(option, dict):
            return option
        raise ValueError(f"an option must be a string or an object, got {option!r}")

    @classmethod
    def _option_text(cls, option: Any) -> str:
        """The display text of a unified option: its `text`, else its `key`."""
        d = cls._as_option_dict(option)
        text = d.get("text")
        return str(text if text is not None else d.get("key", ""))

    @classmethod
    def _option_key(cls, option: Any, index: int) -> str:
        """The answer key of a unified option: its `key`, else its `text`, else `option_{index}`."""
        d = cls._as_option_dict(option)
        key = d.get("key") or d.get("text")
        return str(key) if key is not None else f"option_{index}"


class JevPreprocessor(ScoringPreprocessor):
    """JEV distill-corpus preprocessor: fan a record out into ONE row per question"""

    KIND_KEYS = ("kind", "type", "question_type")
    QUESTION_KEYS = ("question", "prompt", "query")
    OPTION_KEYS = ("options", "opts", "choices")
    STATE_KEYS = ("state", "context", "passage", "text")
    GOLD_KEYS = ("target_probs", "target", "probs", "dist", "distribution", "gold", "label", "answer")
    # Optional: a record that already packs a list of question dicts under one column.
    QUESTIONS_KEYS = ("questions",)

    MODEL_NAME = "JEV"

    fan_out = True

    def _resolve_options(self, question: Dict[str, Any]) -> List[str]:
        canonical = self.default_options(question["kind"])
        if canonical is not None:
            return canonical
        return super()._resolve_options(question)

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default="")
        return state, self._parse_questions(row)

    def _parse_questions(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        packed = self._first(row, self.QUESTIONS_KEYS, default=None)
        if isinstance(packed, (list, tuple)) and packed and isinstance(packed[0], dict):
            return [self._normalize_question(q) for q in packed]
        kind = self._first(row, self.KIND_KEYS, default=None)
        question = self._first(row, self.QUESTION_KEYS, default=None)
        if kind is None or question is None:
            return []
        return [self._normalize_question(row)]

    def _normalize_question(self, source: Dict[str, Any]) -> Dict[str, Any]:
        kind = self._first(source, self.KIND_KEYS, default=None)
        question = self._first(source, self.QUESTION_KEYS, default=None)
        if kind is None or question is None:
            raise ValueError(f"{self.MODEL_NAME} question needs a kind and a question text; got {source!r}")
        out: Dict[str, Any] = {"kind": self._normalize_kind(kind), "question": question}
        options = self._first(source, self.OPTION_KEYS, default=None)
        if options is not None:
            out["options"] = [self._option_text(o) for o in options]
        gold = self._first(source, self.GOLD_KEYS, default=None)
        if gold is not None:
            out["gold"] = gold
        return out


class ClefPreprocessor(ScoringPreprocessor):
    """Clef joint-schema preprocessor: ONE record -> ONE row carrying ALL its questions"""

    STATE_KEYS = ("state", "context", "input")
    QUESTIONS_KEYS = ("questions", "fields", "schema")
    KIND_KEYS = ("type", "kind", "question_type")
    INSTRUCTION_KEYS = ("instructions", "instruction", "question", "prompt", "text")
    CRITERIA_KEYS = ("criteria", "options", "choices", "levels")
    GOLD_KEYS = ("answer", "gold", "label", "target", "value")
    ANSWERS_KEYS = ("answers", "labels", "golds", "targets")
    MODEL_NAME = "Clef"
    # Clef's wire JSON is key-sorted, so a dict state's field order never reaches the row:
    # ``shuffle_fields`` skips the state side (see ``augment_decision_row``).
    state_field_order_fixed = True

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default="")
        answers = self._first(row, self.ANSWERS_KEYS, default=None)
        questions_raw = self._first(row, self.QUESTIONS_KEYS, default=None)
        items = self._question_items(questions_raw)
        questions: List[Dict[str, Any]] = []
        for qid, q in items:
            if not isinstance(q, dict):
                raise ValueError(
                    f"{self.MODEL_NAME} question {qid!r} must be a dict of type/instructions/criteria; got {q!r}"
                )
            kind = self._normalize_kind(self._first(q, self.KIND_KEYS, default=None))
            criteria = self._normalize_criteria(kind, self._first(q, self.CRITERIA_KEYS, default=None))
            instructions = self._first(q, self.INSTRUCTION_KEYS, default=None)
            option_ids = self._option_ids(kind, criteria)
            gold = self._resolve_gold(q, answers, qid, option_ids)
            questions.append(
                {
                    "kind": kind,
                    "question": self._question_text(instructions, qid),
                    "options": option_ids,
                    "question_id": str(qid),
                    "criteria": criteria,
                }
            )
            if gold is not None:
                questions[-1]["gold"] = gold
        return state, questions

    @classmethod
    def _option_ids(cls, kind: str, criteria: Any) -> List[str]:
        """Answer-code ids of a question in prompt order; the subclass's head defines that order"""
        from swift.dev.model.decision_head import clef_option_ids

        return clef_option_ids(kind, criteria)

    def _question_text(self, instructions: Optional[str], qid: Any) -> Optional[str]:
        """Question text; Clef falls back to the question id when a question carries no instructions"""
        return instructions if instructions is not None else str(qid)

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = super()._build_row(row, state, questions)
        if isinstance(state, (str, type(None))):
            out["state"] = state or ""
        else:
            out["state"] = self._encode_json(state)
        out["question_ids"] = [q["question_id"] for q in questions]
        out["criteria"] = [
            self._encode_json(c) if c is not None and not isinstance(c, str) else c
            for c in [q.get("criteria") for q in questions]
        ]
        return out

    @staticmethod
    def _encode_json(value: Any) -> str:
        """How `state`/`criteria` ride the row: Clef's wire JSON is compact and key-sorted, matching
        ``clef_option_ids``' sorted choice order"""
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _state_to_text(state: Any) -> str:
        if isinstance(state, str):
            return state
        if state is None:
            return ""
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _normalize_criteria(cls, kind: str, raw: Any) -> Any:
        """Coerce the unified `options` list into Clef's native `criteria` shape"""
        if raw is None or isinstance(raw, dict):
            return raw
        if not isinstance(raw, (list, tuple)):
            return raw
        if kind == "score":
            return [cls._option_text(o) for o in raw]  # level descriptions
        return {cls._option_key(o, i): cls._option_text(o) for i, o in enumerate(raw)}

    def _resolve_gold(self, question: Dict[str, Any], answers: Any, qid: Any, option_ids: List[str]) -> Optional[Any]:
        gold = self._lookup_gold(question, answers, qid)
        if gold is None:
            return None
        if isinstance(gold, (list, tuple)):
            return [float(x) for x in gold]
        if isinstance(gold, bool):
            gold = "true" if gold else "false"
        if isinstance(gold, int):
            return gold
        text = str(gold).strip()
        for index, option_id in enumerate(option_ids):
            if text == option_id or text.lower() == option_id.lower():
                return index
        if text.lstrip("-").isdigit():
            return int(text)
        raise ValueError(f"{self.MODEL_NAME} gold {gold!r} for question {qid!r} matches no option id {option_ids}.")


class Thx02Preprocessor(ClefPreprocessor):
    MODEL_NAME = "THX-02"
    # THX-02's state column keeps insertion order (see ``_encode_json``), so field order DOES
    # reach the row -- re-enable the state-side shuffle that Clef's fixed order disables.
    state_field_order_fixed = False

    @classmethod
    def _option_ids(cls, kind: str, criteria: Any) -> List[str]:
        from swift.dev.model.decision_head import thx02_option_ids

        return thx02_option_ids(kind, criteria)

    def _question_text(self, instructions: Optional[str], qid: Any) -> Optional[str]:
        # No str(qid) fallback: a question without instructions keeps None, and the template
        # renders its own default instruction (see Thx02Template._encode).
        return instructions

    @staticmethod
    def _encode_json(value: Any) -> str:
        # Default separators and INSERTION ORDER (no sort_keys): the template decodes criteria in
        # this order and zips them with the answer codes, which thx02_option_ids also takes in
        # insertion order -- sorting here would desynchronise codes from options.
        return json.dumps(value, ensure_ascii=False)

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = super()._build_row(row, state, questions)
        out["state_is_json"] = not isinstance(state, str) and state is not None
        return out


def _omnijev_render_option(o: Dict[str, Any]) -> str:
    if o.get("abstain"):
        return "none of the above"
    if "region" in o:
        box = o["region"]["box"]
        b = [int(round(float(v))) for v in box]
        return "region [{},{},{},{}]".format(*b)
    return str(o.get("text", ""))[:200]


def _omnijev_options(q: Dict[str, Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    kind = q["kind"]
    if kind == "noul":
        opts = [{"region": q["region"]}] if q.get("region") else [{"text": ""}]
        return ["yes"], opts
    if kind == "score":
        lv = q.get("levels") or list((q.get("criteria") or {}).keys())
        if not lv and q.get("options") is not None:
            lv = [o if isinstance(o, str) else (o.get("text") or o.get("key")) for o in q["options"]]
        return list(lv), [{"text": str(x)} for x in lv]
    if q.get("options") is not None:
        keys: List[Any] = []
        opts = []
        for i, o in enumerate(q["options"]):
            if isinstance(o, str):
                o = {"text": o}
            if o.get("abstain"):
                continue
            keys.append(o.get("key") or o.get("text") or f"option_{i}")
            opts.append(o)
        return keys, opts
    crit = q.get("criteria") or {}
    keys = list(crit.keys())
    return keys, [{"text": k if crit[k] in (None, "") else f"{k}: {crit[k]}"} for k in keys]


class OmniJevPreprocessor(ScoringPreprocessor):
    STATE_KEYS = ("state", "context", "input")
    QUESTIONS_KEYS = ("questions", "fields")
    KIND_KEYS = ("type", "kind", "question_type")
    INSTRUCTION_KEYS = ("instructions", "instruction", "question", "prompt", "text")
    LEVELS_KEYS = ("levels",)
    CRITERIA_KEYS = ("criteria",)
    OPTIONS_KEYS = ("options",)
    REGION_KEYS = ("region",)
    GOLD_KEYS = ("answer", "gold", "label", "target", "value")
    ANSWERS_KEYS = ("answers", "labels", "golds", "targets")
    IMAGES_KEYS = ("images", "image")
    VIDEOS_KEYS = ("videos", "video")

    MODEL_NAME = "OmniJev"

    fan_out = True

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default=None)
        answers = self._first(row, self.ANSWERS_KEYS, default=None)
        questions_raw = self._first(row, self.QUESTIONS_KEYS, default=None)
        questions: List[Dict[str, Any]] = []
        for qid, q in self._question_items(questions_raw):
            if not isinstance(q, dict):
                raise ValueError(
                    f"{self.MODEL_NAME} question {qid!r} must be a dict of type/instructions/...; got {q!r}"
                )
            kind = self._normalize_kind(self._first(q, self.KIND_KEYS, default=None))
            instruction = self._first(q, self.INSTRUCTION_KEYS, default=None)
            instruction = str(qid) if instruction is None else instruction
            src: Dict[str, Any] = {"kind": kind}
            for keys, dst in (
                (self.LEVELS_KEYS, "levels"),
                (self.CRITERIA_KEYS, "criteria"),
                (self.OPTIONS_KEYS, "options"),
                (self.REGION_KEYS, "region"),
            ):
                value = self._first(q, keys, default=None)
                if value is not None:
                    src[dst] = value
            keys, opts = _omnijev_options(src)
            from swift.dev.model.decision_head import omnijev_column_keys

            column_keys = omnijev_column_keys(kind, keys)
            gold = self._resolve_gold(q, answers, qid, kind, column_keys)
            item: Dict[str, Any] = {
                "kind": kind,
                "question": instruction,
                "options": [_omnijev_render_option(o) for o in opts],
                "column_keys": column_keys,
            }
            if gold is not None:
                item["gold"] = gold
            questions.append(item)
        return state, questions

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        q = questions[0]
        column_keys = q["column_keys"]
        out: Dict[str, Any] = {
            "messages": [{"role": "user", "content": self._content_text(state, q["question"])}],
            "questions": [q["question"]],
            "kinds": [q["kind"]],
            "options": [q["options"]],
            "omnijev_column_keys": [column_keys],
        }
        images, video = self._resolve_media(row, state)
        if images is not None:
            out["images"] = images
        if video is not None:
            # The `video` DESCRIPTOR dict, NOT a swift video file
            out["omnijev_video"] = video
        if q.get("gold") is not None:
            out["target_probs"] = [self.to_target_dist(q["gold"], len(column_keys))]
        return out

    @staticmethod
    def _content_text(state: Any, instruction: str) -> str:
        if isinstance(state, str) and state.strip():
            return f"{state}\n{instruction}"
        if isinstance(state, dict):
            parts = []
            for v in state.values():
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            if parts:
                state_str = "\n".join(parts)
                return f"{state_str}\n{instruction}"
        return instruction

    def _resolve_media(self, row: Dict[str, Any], state: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Pull the OmniJev state's media"""
        state_dict = state if isinstance(state, dict) else {}
        images = self._first(row, self.IMAGES_KEYS, default=None)
        if images is None:
            images = self._first(state_dict, self.IMAGES_KEYS, default=None)
        video = self._first(row, self.VIDEOS_KEYS, default=None)
        if video is None:
            video = self._first(state_dict, self.VIDEOS_KEYS, default=None)
        if isinstance(video, dict):
            return images, video  # the pre-composed mosaic descriptor
        if video is not None:
            raise ValueError(
                f"{self.MODEL_NAME} does not decode a raw video file inline; pre-compose it offline "
                'with `swift.model.omnijev_media.video_state(clip)` -> {"images": [mosaic], "video": {...}} '
                f"and pass that state (got video={video!r})."
            )
        return images, None

    def _resolve_gold(  # noqa: C901
        self, question: Dict[str, Any], answers: Any, qid: Any, kind: str, column_keys: List[str]
    ) -> Optional[Any]:
        gold = self._lookup_gold(question, answers, qid)
        if gold is None:
            return None
        if isinstance(gold, (list, tuple)):
            return [float(x) for x in gold]
        if kind == "noul":
            if isinstance(gold, bool):
                return 0 if gold else 1
            text = str(gold).strip().lower()
            if text in ("yes", "true", "y", "1"):
                return 0
            if text in ("no", "false", "n", "0"):
                return 1
            raise ValueError(f"{self.MODEL_NAME} noul gold {gold!r} for question {qid!r} is not a yes/no value.")
        text = str(gold).strip()
        if text.lower() in ("abstain", "none", "none of the above") and kind == "choice":
            return len(column_keys) - 1
        for index, key in enumerate(column_keys):
            if text == key or text.lower() == key.lower():
                return index
        if text.lstrip("-").isdigit():
            idx = int(text)
            if 0 <= idx < len(column_keys):
                return idx
        raise ValueError(f"{self.MODEL_NAME} gold {gold!r} for question {qid!r} matches no column key {column_keys}.")


class KevPreprocessor(ScoringPreprocessor):
    STATE_KEYS = ("state", "context", "passage", "text", "input")
    QUESTIONS_KEYS = ("questions",)
    KIND_KEYS = ("kind", "type", "question_type")
    INSTR_KEYS = ("instr", "instruction", "instructions", "question", "prompt", "query")
    OPTIONS_KEYS = ("options", "opts", "choices")
    GOLD_KEYS = ("label", "gold", "answer", "target", "value")
    ANSWERS_KEYS = ("answers", "labels", "golds", "targets")

    _NOUL_TEXTS = {"true", "false", "yes", "no", "y", "n"}
    MODEL_NAME = "Kev"

    fan_out = True

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default="")
        answers = self._first(row, self.ANSWERS_KEYS, default=None)
        items = self._question_items(self._first(row, self.QUESTIONS_KEYS, default=None))
        if not items:
            if (
                self._first(row, self.KIND_KEYS, default=None) is None
                and self._first(row, self.INSTR_KEYS, default=None) is None
            ):
                return state, []
            items = [(None, row)]
        questions: List[Dict[str, Any]] = []
        for qid, q in items:
            if not isinstance(q, dict):
                raise ValueError(
                    f"{self.MODEL_NAME} question {qid!r} must be an object with kind/instructions/options, got {{q!r}}."
                )
            options = self._first(q, self.OPTIONS_KEYS, default=None)
            kind = self._first(q, self.KIND_KEYS, default=None)
            if kind is None:
                kind = self._infer_kind(options)
            kind = self._normalize_kind(kind)
            options = self._finalize_options(q, kind, options)
            instr = self._first(q, self.INSTR_KEYS, default=None)
            if instr is None:
                if qid is None:
                    # Single-question fallback form: `qid` is None, so str(qid) would silently
                    # produce the literal 'None' as the question text.
                    raise ValueError(
                        f"{self.MODEL_NAME} single-question row must provide its question text "
                        f"via one of {self.INSTR_KEYS}; got row {q!r}."
                    )
                instr = str(qid)
            item: Dict[str, Any] = {"kind": kind, "question": instr, "options": options}
            gold = self._question_gold(q, answers, qid, kind, options)
            if gold is not None:
                item["gold"] = gold
            questions.append(item)
        return state, questions

    def _finalize_options(self, q: Dict[str, Any], kind: str, options: Optional[List[str]]) -> List[str]:
        if options is not None:
            return [str(o) for o in options]
        default = self.default_options(kind)
        if default is None:
            raise ValueError(f"A 'choice' question must provide its options: {q}")
        return default

    def _question_gold(self, q: Dict[str, Any], answers: Any, qid: Any, kind: str, options: List[str]) -> Optional[Any]:
        return self._resolve_gold(self._lookup_gold(q, answers, qid), qid, kind, options)

    @classmethod
    def _infer_kind(cls, options: Optional[List[str]]) -> str:
        if options and len(options) == 2 and {str(o).strip().lower() for o in options} <= cls._NOUL_TEXTS:
            return "noul"
        if options and all(str(o).strip().lstrip("-").isdigit() for o in options):
            return "score"
        return "choice"

    def _resolve_gold(self, gold: Any, qid: Any, kind: str, options: List[str]) -> Optional[Any]:
        if gold is None:
            return None
        if isinstance(gold, (list, tuple)):
            return [float(x) for x in gold]
        if isinstance(gold, bool):
            if kind != "noul":
                raise ValueError(
                    f"{self.MODEL_NAME} boolean gold {gold!r} for non-noul question {qid!r} (kind={kind})."
                )
            return 1 if gold else 0
        if isinstance(gold, int):
            return gold
        text = str(gold).strip()
        for index, option in enumerate(options):
            if text == option or text.lower() == option.lower():
                return index
        if text.lstrip("-").isdigit():
            return int(text)
        raise ValueError(f"{self.MODEL_NAME} gold {gold!r} for question {qid!r} matches no option {options}.")


class GenericDecisionPreprocessor(JevPreprocessor):
    """Generic decision preprocessor"""

    pass


class LayaPreprocessor(KevPreprocessor):
    QUESTIONS_KEYS = ("qs",) + KevPreprocessor.QUESTIONS_KEYS + ("fields",)
    KIND_KEYS = ("t",) + KevPreprocessor.KIND_KEYS
    INSTR_KEYS = ("ins",) + KevPreprocessor.INSTR_KEYS
    CRITERIA_KEYS = ("crit", "criteria", "levels")
    GOLD_KEYS = ("y", "gold", "label", "answer", "target", "value")
    SOFT_KEYS = ("soft", "target_probs", "probs", "dist")
    MODEL_NAME = "Laya"

    def _finalize_options(self, q: Dict[str, Any], kind: str, options: Optional[List[str]]) -> List[str]:
        return self._laya_options(kind, self._first(q, self.CRITERIA_KEYS, default=None), options)

    def _question_gold(self, q: Dict[str, Any], answers: Any, qid: Any, kind: str, options: List[str]) -> Optional[Any]:
        gold = self._first(q, self.SOFT_KEYS, default=None)
        if gold is None:
            gold = self._resolve_gold(self._lookup_gold(q, answers, qid), qid, kind, options)
        return gold

    def _laya_options(self, kind: str, criteria: Any, options: Optional[List[str]]) -> List[str]:
        if options:
            return [str(o) for o in options]
        if kind == "noul":
            crit = criteria if isinstance(criteria, dict) else {}
            return [
                "false: " + str(crit.get("false") or "no, the statement does not hold"),
                "true: " + str(crit.get("true") or "yes, the statement holds"),
            ]
        if kind == "choice":
            crit = criteria or {}
            if not isinstance(crit, dict):
                raise ValueError(f"Laya choice 'crit' must be a dict {{option_id: criteria}}; got {crit!r}")
            return [str(k) if not v else f"{k}: {v}" for k, v in crit.items()]
        crit = criteria if criteria is not None else list(range(6))
        if not isinstance(crit, (list, tuple)):
            raise ValueError(f"Laya score 'crit' must be a list of level descriptions; got {crit!r}")
        return [f"level {i}: {c}" for i, c in enumerate(crit)]

    @classmethod
    def _infer_kind(cls, options: Optional[List[str]]) -> str:
        if options and len(options) == 2 and {str(o).strip().lower() for o in options} <= cls._NOUL_TEXTS:
            return "noul"
        return "choice"


class StartLuxPreprocessor(ScoringPreprocessor):
    STATE_KEYS = ("state", "context", "input")
    QUESTIONS_KEYS = ("questions", "fields")
    KIND_KEYS = ("type", "kind", "question_type")
    INSTRUCTION_KEYS = ("instructions", "instruction", "question", "prompt", "text")
    CRITERIA_KEYS = ("criteria", "options", "choices", "levels")
    GOLD_KEYS = ("answer", "gold", "label", "target", "value")
    ANSWERS_KEYS = ("answers", "labels", "golds", "targets")
    IMAGES_KEYS = ("images", "image")
    MODEL_NAME = "StartLux"
    _MAX_OPTIONS = 26
    _MAX_LEVELS = 10
    _ANNOTATE_MIN = 8
    _NOUL_INSTRUCTIONS = "Which answer fits the evidence?"
    _NONE_OPTION = "None of the above"

    fan_out = True

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default="")
        answers = self._first(row, self.ANSWERS_KEYS, default=None)
        questions: List[Dict[str, Any]] = []
        for qid, q in self._question_items(self._first(row, self.QUESTIONS_KEYS, default=None)):
            if not isinstance(q, dict):
                raise ValueError(
                    f"{self.MODEL_NAME} question {qid!r} must be an object with "
                    "type/instructions/criteria, got {q!r}."
                )
            kind = self._normalize_kind(self._first(q, self.KIND_KEYS, default=None))
            instructions = self._startlux_value_text(self._first(q, self.INSTRUCTION_KEYS, default=None))
            criteria_raw = self._first(q, self.CRITERIA_KEYS, default=None)
            ids, criteria, none_option = self._startlux_options(kind, criteria_raw, q)
            if not instructions:
                if kind == "noul":
                    instructions = self._NOUL_INSTRUCTIONS
                else:
                    raise ValueError(
                        f"{self.MODEL_NAME} question {qid!r} needs instructions (choice and score "
                        "questions require them)."
                    )
            item: Dict[str, Any] = {
                "kind": kind,
                "question": instructions,
                "options": list(zip(ids, [criteria[i] for i in ids])),
            }
            if none_option:
                item["none_option"] = True
            gold = self._resolve_gold(q, answers, qid, kind, ids)
            if gold is not None:
                item["gold"] = gold
            questions.append(item)
        return state, questions

    @staticmethod
    def _startlux_value_text(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        s = s.strip()
        return s or None

    @classmethod
    def _startlux_annotate_indices(cls, x: Any) -> Any:
        if isinstance(x, list):
            if len(x) >= cls._ANNOTATE_MIN:
                return [
                    (
                        {"_index": i, **cls._startlux_annotate_indices(v)}
                        if isinstance(v, dict)
                        else {"_index": i, "value": cls._startlux_annotate_indices(v)}
                    )
                    for i, v in enumerate(x)
                ]
            return [cls._startlux_annotate_indices(v) for v in x]
        if isinstance(x, dict):
            return {k: cls._startlux_annotate_indices(v) for k, v in x.items()}
        return x

    @classmethod
    def _startlux_state_text(cls, state: Any) -> str:
        if isinstance(state, str):
            return state
        if state is None or state == {} or state == []:
            return ""
        return json.dumps(cls._startlux_annotate_indices(state), ensure_ascii=False)

    def _startlux_options(  # noqa: C901
        self, kind: str, criteria_raw: Any, q: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any], bool]:
        none_option = bool(q.get("allow_none")) and kind == "choice"
        if kind == "noul":
            crit = criteria_raw if isinstance(criteria_raw, dict) else {}
            ids = ["true", "false"]
            criteria = {i: self._startlux_value_text(crit.get(i)) for i in ids}
            return ids, criteria, False
        if kind == "score":
            if isinstance(criteria_raw, dict):
                try:
                    keys = sorted(criteria_raw, key=lambda k: float(k))
                except (TypeError, ValueError):
                    keys = list(criteria_raw)
                levels = [criteria_raw[k] for k in keys]
            elif isinstance(criteria_raw, (list, tuple)):
                levels = list(criteria_raw)
            else:
                raise ValueError(f"{self.MODEL_NAME} score question needs criteria: a list of levels, lowest first.")
            if not 2 <= len(levels) <= self._MAX_LEVELS:
                raise ValueError(
                    f"{self.MODEL_NAME} score question needs 2 to {self._MAX_LEVELS} levels, got {len(levels)}."
                )
            ids = [str(i) for i in range(len(levels))]
            return ids, {i: self._startlux_value_text(c) for i, c in zip(ids, levels)}, False
        if isinstance(criteria_raw, (list, tuple)):
            criteria_raw = {self._option_key(o, i): self._option_text(o) for i, o in enumerate(criteria_raw)}
        if not isinstance(criteria_raw, dict):
            raise ValueError(
                f"{self.MODEL_NAME} choice question needs criteria: {{option: description or null}} "
                "or a list of options."
            )
        ids = [str(k) for k in criteria_raw.keys()]
        criteria = {i: self._startlux_value_text(criteria_raw[i]) for i in ids}
        if none_option:
            if any(_startlux_norm(i) == _startlux_norm(self._NONE_OPTION) for i in ids):
                raise ValueError(
                    f'{self.MODEL_NAME} allow_none adds the option "{self._NONE_OPTION}" itself: '
                    "remove the option of that name or leave allow_none out."
                )
            ids.append(self._NONE_OPTION)
            criteria[self._NONE_OPTION] = None
        if not 2 <= len(ids) <= self._MAX_OPTIONS:
            raise ValueError(
                f"{self.MODEL_NAME} choice question needs 2 to {self._MAX_OPTIONS} options in one "
                f"pass, got {len(ids)}; lists over {self._MAX_OPTIONS - 1} go through the official "
                "server's knockout rounds, which swift does not implement."
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.MODEL_NAME} duplicate option id in {ids}.")
        for i in ids:
            if not i.strip() or "\n" in i or "\r" in i or len(i) > 200:
                raise ValueError(
                    f"{self.MODEL_NAME} bad option id {i!r}: ids must be non-empty single-line strings "
                    "of at most 200 characters."
                )
        return ids, criteria, none_option

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        q = questions[0]
        ids = [i for i, _ in q["options"]]
        criteria = dict(q["options"])
        state_text = self._startlux_state_text(state)
        out: Dict[str, Any] = {
            "messages": [{"role": "user", "content": state_text}],
            "questions": [q["question"]],
            "kinds": [q["kind"]],
            "options": [ids],
            "criterias": [[criteria[i] for i in ids]],
            "none_options": [bool(q.get("none_option")) and ids[-1:] == [self._NONE_OPTION]],
            "state": state_text,
        }
        images = self._first(row, self.IMAGES_KEYS, default=None)
        if images is not None:
            out["images"] = images if isinstance(images, (list, tuple)) else [images]
        if q.get("gold") is not None:
            out["target_probs"] = [self.to_target_dist(q["gold"], len(ids))]
        return out

    def _resolve_gold(  # noqa: C901
        self, question: Dict[str, Any], answers: Any, qid: Any, kind: str, option_ids: List[str]
    ) -> Optional[Any]:
        gold = self._lookup_gold(question, answers, qid)
        if gold is None:
            return None
        if isinstance(gold, (list, tuple)):
            return [float(x) for x in gold]
        if isinstance(gold, bool):
            if kind != "noul":
                raise ValueError(f"{self.MODEL_NAME} boolean gold {gold!r} for non-noul question {qid!r}.")
            return 0 if gold else 1
        if isinstance(gold, int):
            if not 0 <= gold < len(option_ids):
                raise ValueError(f"{self.MODEL_NAME} gold index {gold} out of range for {option_ids}.")
            return gold
        text = str(gold).strip()
        if kind == "noul" and text.lower() in ("yes", "y"):
            return 0
        if kind == "noul" and text.lower() in ("no", "n"):
            return 1
        if text.lower() in ("none", "none of the above") and option_ids[-1] == self._NONE_OPTION:
            return len(option_ids) - 1
        for index, option_id in enumerate(option_ids):
            if text == option_id or text.lower() == option_id.lower():
                return index
        if text.lstrip("-").isdigit():
            idx = int(text)
            if 0 <= idx < len(option_ids):
                return idx
        raise ValueError(f"{self.MODEL_NAME} gold {gold!r} for question {qid!r} matches no option id {option_ids}.")
