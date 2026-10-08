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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

from .core import RowPreprocessor


class ScoringPreprocessor(RowPreprocessor):
    """Base: assemble the decision row contract; subclasses supply `parse_record`."""

    NOUL_OPTIONS = ['false', 'true']
    SCORE_OPTIONS = ['0', '1', '2', '3', '4', '5']

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state, questions = self.parse_record(row)
        if not questions:
            return None
        return self._build_row(row, state, questions)

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble ONE decision row (the contract in the module docstring) from `(state, questions)`.

        Split out of `preprocess` so a subclass can fan a record out into several rows (JEV renders one
        question per prompt) while reusing this row-building logic verbatim.
        """
        kinds = [q['kind'] for q in questions]
        options = [self._resolve_options(q) for q in questions]
        out: Dict[str, Any] = {
            'messages': [{'role': 'user', 'content': state if state is not None else ''}],
            'questions': [q['question'] for q in questions],
            'kinds': kinds,
            'options': options,
        }
        for key in ('images', 'videos', 'audios'):
            if row.get(key) is not None:
                out[key] = row[key]
        if self._has_gold(questions):
            out['target_probs'] = [self.to_target_dist(q['gold'], len(opt)) for q, opt in zip(questions, options)]
        return out

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        """Return `(state, questions)`; each question is `{'kind','question','options','gold'}`.

        `options` may be omitted for noul/score (defaults are filled by `_resolve_options`); `gold`
        may be None at inference. Must be implemented by each model subclass.
        """
        raise NotImplementedError

    def _resolve_options(self, question: Dict[str, Any]) -> List[str]:
        options = question.get('options')
        if options:
            return list(options)
        default = self.default_options(question['kind'])
        if default is None:
            raise ValueError(f"A 'choice' question must provide its options: {question}")
        return default

    @classmethod
    def default_options(cls, kind: str) -> Optional[List[str]]:
        if kind == 'noul':
            return list(cls.NOUL_OPTIONS)
        if kind == 'score':
            return list(cls.SCORE_OPTIONS)
        return None

    @staticmethod
    def _has_gold(questions: Sequence[Dict[str, Any]]) -> bool:
        return any(q.get('gold') is not None for q in questions)

    @staticmethod
    def to_target_dist(gold: Any, n_options: int) -> List[float]:
        """Gold -> a distribution over the question's own `n_options` options.

        An int gold is the correct option index (one-hot); a sequence gold is already a full
        distribution (JEV distillation), and must match `n_options`.
        """
        if isinstance(gold, (list, tuple)):
            probs = [float(x) for x in gold]
            if len(probs) != n_options:
                raise ValueError(f'target distribution length {len(probs)} != n_options {n_options}')
            return probs
        idx = int(gold)
        if not 0 <= idx < n_options:
            raise ValueError(f'gold index {idx} out of range for n_options {n_options}')
        dist = [0.0] * n_options
        dist[idx] = 1.0
        return dist

    @staticmethod
    def _first(source: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
        """Return the first non-None value among `keys` (a permissive column lookup shared by the
        model subclasses, whose datasets use differing column names)."""
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
        return default

    @staticmethod
    def _as_option_dict(option: Any) -> Dict[str, Any]:
        """Normalize ONE unified option element to a dict.

        The user-facing schema lets an option be either a plain string (sugar for `{'text': s}`) or
        an object `{'key'?, 'text'?, 'abstain'?, 'region'?}`. Each subclass reads only the fields it
        needs (JEV: text; Clef: key + text; OmniJev: key/text/abstain/region), so all three accept
        the SAME option representation.
        """
        if isinstance(option, str):
            return {'text': option}
        if isinstance(option, dict):
            return option
        raise ValueError(f'an option must be a string or an object, got {option!r}')

    @classmethod
    def _option_text(cls, option: Any) -> str:
        """The display text of a unified option: its `text`, else its `key`."""
        d = cls._as_option_dict(option)
        text = d.get('text')
        return str(text if text is not None else d.get('key', ''))

    @classmethod
    def _option_key(cls, option: Any, index: int) -> str:
        """The answer key of a unified option: its `key`, else its `text`, else `option_{index}`."""
        d = cls._as_option_dict(option)
        key = d.get('key') or d.get('text')
        return str(key) if key is not None else f'option_{index}'


class JevPreprocessor(ScoringPreprocessor):
    """JEV distill-corpus preprocessor: fan a record out into ONE row per question.

    JEV's System-1 pass (`serve_decide.py::s1_pass`) scores a single question per forward, so a record
    carrying several questions becomes several rows (`RowPreprocessor.batched_preprocess` accepts a
    list return, core.py:214-220), each rendering one bare prompt and carrying `counts=[1]` after the
    collator. This matches s1_pass exactly (one CompletionRequest per question).

    `parse_record` maps the corpus columns onto the shared question dict `{kind, question, options,
    gold}`. `gold` is the teacher's soft distribution over the question's OWN options (JEV
    distillation) or an option index; `to_target_dist` validates its length against the option count.

    NOTE(schema): `SargeDev/jev-distill-corpus-v3`'s exact column names could not be verified offline
    (HF page/API unreachable in this environment), so the field lookup below is intentionally
    permissive (it tries the natural serve_decide names plus common aliases). It MUST be re-checked
    against the real dataset before a training run; a `columns={...}` remap can be passed to
    `__init__` to rename the corpus columns onto the canonical names used here.
    """

    KIND_KEYS = ('kind', 'type', 'question_type')
    QUESTION_KEYS = ('question', 'prompt', 'query')
    OPTION_KEYS = ('options', 'opts', 'choices')
    STATE_KEYS = ('state', 'context', 'passage', 'text')
    GOLD_KEYS = ('target_probs', 'target', 'probs', 'dist', 'distribution', 'gold', 'label', 'answer')
    # Optional: a record that already packs a list of question dicts under one column.
    QUESTIONS_KEYS = ('questions',)
    KIND_ALIASES = {
        'noul': 'noul',
        'yes_no': 'noul',
        'yn': 'noul',
        'bool': 'noul',
        'choice': 'choice',
        'multiple_choice': 'choice',
        'mcq': 'choice',
        'select': 'choice',
        'score': 'score',
        'rating': 'score',
        'ordinal': 'score',
        'likert': 'score',
    }

    def preprocess(self, row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        state, questions = self.parse_record(row)
        if not questions:
            return None
        # One row per question (JEV renders a single question per prompt).
        return [self._build_row(row, state, [q]) for q in questions]

    def _resolve_options(self, question: Dict[str, Any]) -> List[str]:
        """Force the canonical option set for noul/score; only `choice` uses caller-supplied options.

        JEV's System-1 head scores a FIXED per-kind verbalizer: noul always reads the two tokens for
        ['false','true'] and score always reads the six digit tokens ['0'..'5'], regardless of any
        option text a record carries. `serve_decide.py::decide()` mirrors this -- it forces
        `['false','true']` / `[str(i) for i in range(6)]` and ignores caller options for those kinds
        (only `choice` consumes them). Rendering custom noul/score text would desync the prompt from
        the tokens the head actually scores and diverge from the oracle, so we force canonical here
        (model-local override; the shared `ScoringPreprocessor` and Clef/OmniJev are unaffected).
        """
        canonical = self.default_options(question['kind'])
        if canonical is not None:  # noul / score
            return canonical
        return super()._resolve_options(question)  # choice: caller options are required

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default='')
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
            raise ValueError(f'JEV question needs a kind and a question text; got {source!r}')
        out: Dict[str, Any] = {'kind': self._normalize_kind(kind), 'question': question}
        options = self._first(source, self.OPTION_KEYS, default=None)
        if options is not None:
            # Unified schema: an option may be a plain string or an object; JEV renders only the
            # display text (no abstain/region), so collapse each element to its text.
            out['options'] = [self._option_text(o) for o in options]
        gold = self._first(source, self.GOLD_KEYS, default=None)
        if gold is not None:
            out['gold'] = gold
        return out

    @classmethod
    def _normalize_kind(cls, kind: Any) -> str:
        key = str(kind).strip().lower()
        if key not in cls.KIND_ALIASES:
            raise ValueError(f'unknown JEV question kind {kind!r}; expected one of noul/choice/score (or an alias).')
        return cls.KIND_ALIASES[key]


def _clef_option_ids(kind: str, criteria: Any) -> List[str]:
    """Clef option ids in the exact order `joint_schema_model.py::question_options` emits them.

    MUST stay identical to `ClefTemplate._question_options` (both mirror the official ordering), so the
    gold one-hot index computed here lines up with the logit column the head produces:
    noul -> ['true','false'], choice -> criteria keys sorted as strings, score -> ['0','1',...].
    """
    if kind == 'noul':
        return ['true', 'false']
    if kind == 'choice':
        return [str(key) for key, _ in sorted((str(k), v) for k, v in (criteria or {}).items())]
    return [str(index) for index in range(len(criteria or []))]


class ClefPreprocessor(ScoringPreprocessor):
    """Clef joint-schema preprocessor: ONE record -> ONE row carrying ALL its questions.

    Clef decodes a record's questions jointly (decision-models-comparison.md §2: Clef is the only
    multi-question-per-forward model), so unlike `JevPreprocessor` this does NOT fan out; the base
    `preprocess` builds a single row whose `kinds` / `questions` / `options` / `target_probs` are
    parallel lists over the record's questions, in the original `questions` order.

    A Clef record is `{id?, state, questions: {qid: {type, instructions, criteria}}, <gold>}` where
    `criteria` is a dict {option_id: description} for `choice`, a list of level descriptions for
    `score`, and optional for `noul`. `state` may be ANY JSON value (rendered verbatim by the
    template), so the raw value is carried in a `state` key and `question_ids` / `criteria` are added
    for `ClefTemplate` to rebuild the byte-exact SCHEMA FIELDS layout.

    NOTE(schema): Clef's training data is closed-source (decision-models-comparison.md §5), so the gold
    column layout could not be verified offline; `_resolve_gold` is intentionally permissive (accepts
    an option id, a bool for noul, an int index, or a full distribution, at question level or in a
    record-level `answers` map) and MUST be re-checked against the real dataset before a training run.
    """

    STATE_KEYS = ('state', 'context', 'input')
    QUESTIONS_KEYS = ('questions', 'fields', 'schema')
    KIND_KEYS = ('type', 'kind', 'question_type')
    INSTRUCTION_KEYS = ('instructions', 'instruction', 'question', 'prompt', 'text')
    CRITERIA_KEYS = ('criteria', 'options', 'choices', 'levels')
    GOLD_KEYS = ('answer', 'gold', 'label', 'target', 'value')
    ANSWERS_KEYS = ('answers', 'labels', 'golds', 'targets')
    ID_KEYS = ('id', 'record_id')
    KIND_ALIASES = JevPreprocessor.KIND_ALIASES

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default='')
        answers = self._first(row, self.ANSWERS_KEYS, default=None)
        questions_raw = self._first(row, self.QUESTIONS_KEYS, default=None)
        items = self._question_items(questions_raw)
        questions: List[Dict[str, Any]] = []
        for qid, q in items:
            if not isinstance(q, dict):
                raise ValueError(f'Clef question {qid!r} must be a dict of type/instructions/criteria; got {q!r}')
            kind = self._normalize_kind(self._first(q, self.KIND_KEYS, default=None))
            criteria = self._normalize_criteria(kind, self._first(q, self.CRITERIA_KEYS, default=None))
            instructions = self._first(q, self.INSTRUCTION_KEYS, default=None)
            option_ids = _clef_option_ids(kind, criteria)
            gold = self._resolve_gold(q, answers, qid, option_ids)
            questions.append({
                'kind': kind,
                'question': instructions if instructions is not None else str(qid),
                'options': option_ids,
                'question_id': str(qid),
                'criteria': criteria,
            })
            if gold is not None:
                questions[-1]['gold'] = gold
        return state, questions

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = super()._build_row(row, state, questions)
        # `state` may be a non-str JSON value; keep the message content a string and carry the raw
        # value separately so ClefTemplate can render it byte-exactly (json for non-str).
        out['messages'] = [{'role': 'user', 'content': self._state_to_text(state)}]
        out['state'] = state
        out['question_ids'] = [q['question_id'] for q in questions]
        out['criteria'] = [q.get('criteria') for q in questions]
        return out

    @staticmethod
    def _state_to_text(state: Any) -> str:
        if isinstance(state, str):
            return state
        if state is None:
            return ''
        return json.dumps(state, ensure_ascii=False, separators=(',', ':'), sort_keys=True)

    @staticmethod
    def _question_items(questions_raw: Any) -> List[Tuple[Any, Any]]:
        if isinstance(questions_raw, dict):
            return list(questions_raw.items())
        if isinstance(questions_raw, (list, tuple)):
            return list(enumerate(questions_raw))
        return []

    @classmethod
    def _normalize_criteria(cls, kind: str, raw: Any) -> Any:
        """Coerce the unified `options` list into Clef's native `criteria` shape.

        Clef renders byte-exact against the official SCHEMA FIELDS layout, which needs a
        `{option_id: description}` map for `choice` (columns sorted by id) and a list of level
        descriptions for `score`. A native `criteria` dict/list is passed through; a unified
        `options` list of strings / `{key,text}` objects is converted to that shape, so the SAME
        user-facing schema works for Clef too.
        """
        if raw is None or isinstance(raw, dict):
            return raw  # native criteria map (choice) or omitted (noul)
        if not isinstance(raw, (list, tuple)):
            return raw
        if kind == 'score':
            return [cls._option_text(o) for o in raw]  # level descriptions
        # choice (noul ignores criteria): {option_id: description}, ids from key/text/index.
        return {cls._option_key(o, i): cls._option_text(o) for i, o in enumerate(raw)}

    def _resolve_gold(self, question: Dict[str, Any], answers: Any, qid: Any,
                      option_ids: List[str]) -> Optional[Any]:
        """Resolve the correct answer to an option INDEX (or a full distribution), or None at inference.

        Accepts, at question level then in a record-level `answers` map: a distribution (list/tuple of
        floats), a bool (noul), an option id string, or an int index. An unmatched value raises rather
        than silently mis-indexing (which would poison the one-hot target).
        """
        gold = self._first(question, self.GOLD_KEYS, default=None)
        if gold is None and isinstance(answers, dict):
            gold = answers.get(str(qid), answers.get(qid))
        if gold is None:
            return None
        if isinstance(gold, (list, tuple)):
            return [float(x) for x in gold]  # a full distribution; to_target_dist validates its length
        if isinstance(gold, bool):
            gold = 'true' if gold else 'false'
        if isinstance(gold, int):
            return gold
        text = str(gold).strip()
        for index, option_id in enumerate(option_ids):
            if text == option_id or text.lower() == option_id.lower():
                return index
        if text.lstrip('-').isdigit():
            return int(text)
        raise ValueError(f'Clef gold {gold!r} for question {qid!r} matches no option id {option_ids}.')

    @classmethod
    def _normalize_kind(cls, kind: Any) -> str:
        key = str(kind).strip().lower()
        if key not in cls.KIND_ALIASES:
            raise ValueError(f'unknown Clef question kind {kind!r}; expected one of noul/choice/score (or an alias).')
        return cls.KIND_ALIASES[key]


def _omnijev_render_option(o: Dict[str, Any]) -> str:
    """Verbatim port of `mso/records.py::render_option`: how one option block is written into the
    prompt. An abstain option renders as 'none of the above'; a region option as its integer box;
    otherwise the option text truncated to 200 chars."""
    if o.get('abstain'):
        return 'none of the above'
    if 'region' in o:
        box = o['region']['box']
        b = [int(round(float(v))) for v in box]
        return 'region [{},{},{},{}]'.format(*b)
    return str(o.get('text', ''))[:200]


def _omnijev_options(q: Dict[str, Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Verbatim port of `mso/infer.py::MSO1._options` -> `(keys, opts)`.

    `keys` are the answer keys (noul -> ['yes'], score -> the level labels, choice -> the option
    keys); `opts` are the option-BLOCK dicts the template renders with `_omnijev_render_option`
    (noul -> ONE block, score -> L blocks, choice -> K blocks, abstain excluded -- it is the head's
    extra learned column, not a rendered block). Accepts both the Jev `criteria`-map shape and the
    OmniJev `options`-list shape (whose entries may be region options).
    """
    kind = q['kind']
    if kind == 'noul':
        opts = [{'region': q['region']}] if q.get('region') else [{'text': ''}]
        return ['yes'], opts
    if kind == 'score':
        lv = q.get('levels') or list((q.get('criteria') or {}).keys())
        if not lv and q.get('options') is not None:
            # Unified schema: levels given as a plain `options` list (strings or {text/key}).
            lv = [o if isinstance(o, str) else (o.get('text') or o.get('key')) for o in q['options']]
        return list(lv), [{'text': str(x)} for x in lv]
    if q.get('options') is not None:
        keys: List[Any] = []
        opts = []
        for i, o in enumerate(q['options']):
            if isinstance(o, str):  # unified sugar: a plain string is {'text': s}
                o = {'text': o}
            if o.get('abstain'):
                continue
            keys.append(o.get('key') or o.get('text') or f'option_{i}')
            opts.append(o)
        return keys, opts
    crit = q.get('criteria') or {}
    keys = list(crit.keys())
    return keys, [{'text': k if crit[k] in (None, '') else f'{k}: {crit[k]}'} for k in keys]


def _omnijev_column_keys(kind: str, keys: Sequence[Any]) -> List[str]:
    """The answer keys of each emitted logit COLUMN, width-matched to `OmniJevHead`'s layout:
    noul -> ['yes','no'] (2 cols; col0 == P(yes)); choice -> keys + ['abstain'] (K+1 cols); score ->
    the level labels (L cols). MUST stay identical to the head's column layout so the gold one-hot
    index computed here lines up with the logit column the head produces."""
    if kind == 'noul':
        return ['yes', 'no']
    if kind == 'choice':
        return [str(k) for k in keys] + ['abstain']
    return [str(k) for k in keys]


class OmniJevPreprocessor(ScoringPreprocessor):
    """OmniJev preprocessor: fan a record out into ONE row per question (like `JevPreprocessor`).

    OmniJev scores a single question per forward (`mso/infer.py::ask`/`ask_branch`; the K option
    rows of decision G1's branch are expanded at forward time, not here), so a record carrying
    several questions becomes several rows. Each row carries the official option-BLOCK structure
    (`_omnijev_options`) plus the width-matched `column_keys`, and the gold as a distribution over
    the COLUMN width (noul 2 / choice K+1 incl. the abstain column / score L) -- NOT the block count,
    because for noul (1 block -> 2 cols) and choice (K blocks -> K+1 cols) they differ.

    The state is multimodal (`{'images': [...], 'video': {...}}` in official `system_one`); the raw
    `images` / `videos` are passed through and the panel/mosaic composition (decision H) is applied
    by `OmniJevTemplate`, not here. The per-question `instructions` become the message text.

    NOTE(schema): OmniJev's training corpus is not published (only the `system_one` request shape and
    the infer CLI's `--questions` map are), so the field lookup below is intentionally permissive and
    MUST be re-checked against the real dataset before a training run; a `columns={...}` remap can be
    passed to `__init__` to rename corpus columns onto the canonical names used here.
    """

    STATE_KEYS = ('state', 'context', 'input')
    QUESTIONS_KEYS = ('questions', 'fields')
    KIND_KEYS = ('type', 'kind', 'question_type')
    INSTRUCTION_KEYS = ('instructions', 'instruction', 'question', 'prompt', 'text')
    LEVELS_KEYS = ('levels',)
    CRITERIA_KEYS = ('criteria',)
    OPTIONS_KEYS = ('options',)
    REGION_KEYS = ('region',)
    GOLD_KEYS = ('answer', 'gold', 'label', 'target', 'value')
    ANSWERS_KEYS = ('answers', 'labels', 'golds', 'targets')
    IMAGES_KEYS = ('images', 'image')
    VIDEOS_KEYS = ('videos', 'video')
    KIND_ALIASES = JevPreprocessor.KIND_ALIASES

    def preprocess(self, row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        state, questions = self.parse_record(row)
        if not questions:
            return None
        return [self._build_row(row, state, [q]) for q in questions]

    def parse_record(self, row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        state = self._first(row, self.STATE_KEYS, default=None)
        answers = self._first(row, self.ANSWERS_KEYS, default=None)
        questions_raw = self._first(row, self.QUESTIONS_KEYS, default=None)
        questions: List[Dict[str, Any]] = []
        for qid, q in self._question_items(questions_raw):
            if not isinstance(q, dict):
                raise ValueError(f'OmniJev question {qid!r} must be a dict of type/instructions/...; got {q!r}')
            kind = self._normalize_kind(self._first(q, self.KIND_KEYS, default=None))
            instruction = self._first(q, self.INSTRUCTION_KEYS, default=None)
            instruction = str(qid) if instruction is None else instruction
            src: Dict[str, Any] = {'kind': kind}
            for keys, dst in ((self.LEVELS_KEYS, 'levels'), (self.CRITERIA_KEYS, 'criteria'),
                              (self.OPTIONS_KEYS, 'options'), (self.REGION_KEYS, 'region')):
                value = self._first(q, keys, default=None)
                if value is not None:
                    src[dst] = value
            keys, opts = _omnijev_options(src)
            column_keys = _omnijev_column_keys(kind, keys)
            gold = self._resolve_gold(q, answers, qid, kind, column_keys)
            item: Dict[str, Any] = {
                'kind': kind,
                'question': instruction,
                'options': [_omnijev_render_option(o) for o in opts],
                'omnijev_options': opts,
                'column_keys': column_keys,
            }
            if gold is not None:
                item['gold'] = gold
            questions.append(item)
        return state, questions

    def _build_row(self, row: Dict[str, Any], state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        q = questions[0]  # fan-out: exactly one question per row
        column_keys = q['column_keys']
        out: Dict[str, Any] = {
            'messages': [{'role': 'user', 'content': self._content_text(state, q['question'])}],
            'questions': [q['question']],
            'kinds': [q['kind']],
            'options': [q['options']],
            'omnijev_options': [q['omnijev_options']],
            'omnijev_column_keys': [column_keys],
        }
        images, video = self._resolve_media(row, state)
        if images is not None:
            out['images'] = images
        if video is not None:
            # The official `video` DESCRIPTOR dict, NOT a swift video file: route it to extra_kwargs so
            # OmniJevTemplate's `video_content` can crop the mosaic back into frames. Putting it in
            # `videos` would make swift treat it as a clip list and prepend a spurious `<video>` tag.
            out['omnijev_video'] = video
        if q.get('gold') is not None:
            out['target_probs'] = [self.to_target_dist(q['gold'], len(column_keys))]
        return out

    @staticmethod
    def _content_text(state: Any, instruction: str) -> str:
        """The message text. Official OmniJev has NO free-text state (the state is the image/video and
        the instruction is the only text), so this is just the instruction; a str `state` (rare, e.g.
        an extra context field) is prepended verbatim so nothing is silently dropped."""
        if isinstance(state, str) and state.strip():
            return f'{state}\n{instruction}'
        return instruction

    def _resolve_media(self, row: Dict[str, Any], state: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Pull the OmniJev state's media: `images` (list of paths/PIL; `images[0]` is the pre-composed
        video mosaic when a `video` descriptor is present) and the official `video` DESCRIPTOR dict
        `{'n_frames','cols','tile','timestamps','duration'}`. That descriptor is metadata for
        `OmniJevTemplate`'s `video_content` crop, NOT a swift video file, so the caller routes it to
        `extra_kwargs['omnijev_video']` and never to `videos` (which swift would treat as a clip list and
        prepend a spurious `<video>` tag). A raw video FILE is not decoded inline -- official runs
        `video_state` offline -- so raise instead of silently mis-encoding. The panel/mosaic composition
        (decision H) is deferred to `OmniJevTemplate`; this only carries the raw media through."""
        state_dict = state if isinstance(state, dict) else {}
        images = self._first(row, self.IMAGES_KEYS, default=None)
        if images is None:
            images = self._first(state_dict, self.IMAGES_KEYS, default=None)
        video = self._first(row, self.VIDEOS_KEYS, default=None)
        if video is None:
            video = self._first(state_dict, self.VIDEOS_KEYS, default=None)
        if isinstance(video, dict):
            return images, video  # the official pre-composed mosaic descriptor
        if video is not None:
            raise ValueError(
                'OmniJev does not decode a raw video file inline; pre-compose it offline with '
                '`swift.model.omnijev_media.video_state(clip)` -> {"images": [mosaic], "video": {...}} '
                f'and pass that state (got video={video!r}).')
        return images, None

    @staticmethod
    def _question_items(questions_raw: Any) -> List[Tuple[Any, Any]]:
        """`mso/records.py::q_items` shape tolerance: a {qid: q} dict or a [q, ...] list."""
        if isinstance(questions_raw, dict):
            return list(questions_raw.items())
        if isinstance(questions_raw, (list, tuple)):
            return list(enumerate(questions_raw))
        return []

    def _resolve_gold(self, question: Dict[str, Any], answers: Any, qid: Any, kind: str,
                      column_keys: List[str]) -> Optional[Any]:
        """Resolve the correct answer to a COLUMN index (or a full distribution over columns), or None
        at inference. noul accepts a bool / 'yes'|'no' / 'true'|'false' / 1|0 (col0 == yes); choice and
        score match the gold text against `column_keys` (choice 'abstain'/'none' -> the abstain column)
        or accept an explicit index. An unmatched value raises rather than silently mis-indexing."""
        gold = self._first(question, self.GOLD_KEYS, default=None)
        if gold is None and isinstance(answers, dict):
            gold = answers.get(str(qid), answers.get(qid))
        if gold is None:
            return None
        if isinstance(gold, (list, tuple)):
            return [float(x) for x in gold]  # a full distribution; to_target_dist validates its width
        if kind == 'noul':
            if isinstance(gold, bool):
                return 0 if gold else 1
            text = str(gold).strip().lower()
            if text in ('yes', 'true', 'y', '1'):
                return 0
            if text in ('no', 'false', 'n', '0'):
                return 1
            raise ValueError(f'OmniJev noul gold {gold!r} for question {qid!r} is not a yes/no value.')
        text = str(gold).strip()
        if text.lower() in ('abstain', 'none', 'none of the above') and kind == 'choice':
            return len(column_keys) - 1
        for index, key in enumerate(column_keys):
            if text == key or text.lower() == key.lower():
                return index
        if text.lstrip('-').isdigit():
            idx = int(text)
            if 0 <= idx < len(column_keys):
                return idx
        raise ValueError(f'OmniJev gold {gold!r} for question {qid!r} matches no column key {column_keys}.')

    @classmethod
    def _normalize_kind(cls, kind: Any) -> str:
        key = str(kind).strip().lower()
        if key not in cls.KIND_ALIASES:
            raise ValueError(
                f'unknown OmniJev question kind {kind!r}; expected one of noul/choice/score (or an alias).')
        return cls.KIND_ALIASES[key]

