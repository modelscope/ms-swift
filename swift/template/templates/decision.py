# Copyright (c) ModelScope Contributors. All rights reserved.
"""Templates for the `decision` task_type (typed-decision System-1 scoring models).

Each decision model renders its own prompt layout and records the per-question locating meta the
scoring head consumes (`decision_meta`, see `swift/model/decision_head.py`). These templates share a
common mixin (`DecisionTemplate`) for the bare-completion prompt layout used by JEV and
GenericDecision; Clef and OmniJev override `_encode`/`_post_encode` entirely because their prompt
assembly and head-meta contracts are fundamentally different.
"""
import json
import string
from typing import Any, Dict, List, Optional, Tuple

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context
from .qwen import Qwen3_5Template, QwenTemplateMeta

# NOTE: `QUESTION_TYPES` lives in `swift.model.decision_head`, but a module-level import here would
# create a swift.template -> swift.model load cycle (swift.model.models.* import swift.template).
# Imported lazily inside the methods that need it.

_OPTION_LETTERS = string.ascii_uppercase


class DecisionTemplate:
    """Shared logic for decision templates that render the bare ``[kind]/[state]/[question]/
    [options]/[decision]:`` completion prompt and attach single-question ``decision_meta``.

    Feeds vLLM a raw ``CompletionRequest`` (profile prefix='', ``add_special_tokens=False``) with
    this exact layout, each field separated by a single newline and the whole prompt ending at
    ``[decision]:`` (no trailing newline)::

        [kind] {kind}
        [state] {state}
        [question] {question}
        [options]
        {option lines}
        [decision]:

    Subclasses set ``_error_prefix`` for model-specific error messages and may override
    ``_max_choice_options`` to cap the number of ``choice`` options.
    """
    _error_prefix: str = 'Decision'
    _max_choice_options: int = len(_OPTION_LETTERS)

    def _render_s1_prompt(self, fields: Dict[str, Any]) -> str:
        kind = fields['kind']
        options: List[str] = fields['options']
        if kind == 'choice':
            if len(options) > self._max_choice_options:
                raise ValueError(f'{self._error_prefix} choice supports at most {self._max_choice_options} options, '
                                 f'got {len(options)}.')
            lines = [f'{_OPTION_LETTERS[i]}) {opt}' for i, opt in enumerate(options)]
        else:
            lines = [str(opt) for opt in options]
        return (f'[kind] {kind}\n[state] {fields["state"]}\n[question] {fields["question"]}\n'
                '[options]\n' + '\n'.join(lines) + '\n[decision]:')

    def _decision_fields(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        extra = inputs.extra_kwargs
        kinds = extra.get('kinds')
        questions = extra.get('questions')
        options = extra.get('options')
        if not kinds or not questions or not options:
            raise ValueError(f'{self._error_prefix}Template requires `kinds` / `questions` / `options` in extra_kwargs '
                             f'(produced by {self._error_prefix}Preprocessor); got '
                             f'kinds={kinds!r}, questions={questions!r}, options={options!r}.')
        kind = self._normalize_kind(kinds[0])
        return {
            'kind': kind,
            'question': questions[0],
            'options': list(options[0]),
            'state': self._get_state(inputs),
        }

    @staticmethod
    def _get_state(inputs: StdTemplateInputs) -> str:
        for message in inputs.messages:
            if message.get('role') == 'user':
                return message.get('content') or ''
        return ''

    @staticmethod
    def _normalize_kind(kind: Any) -> str:
        from swift.model.decision_head import QUESTION_TYPES
        kind = str(kind).strip().lower()
        if kind not in QUESTION_TYPES:
            raise ValueError(f'Question kind must be one of {list(QUESTION_TYPES)}, got {kind!r}.')
        return kind

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        Template._swift_prepare_inputs(self, inputs)

    def _swift_encode(self, inputs: StdTemplateInputs):
        fields = self._decision_fields(inputs)
        content = self._render_s1_prompt(fields)
        # all-zero loss scale: decision has no token-level labels (scored per option).
        return [content], [0.], 0

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # Force the swift backend; never let --template_backend jinja swap in a chat template.
        prev_backend = self.template_backend
        self.template_backend = 'swift'
        try:
            encoded = super()._encode(inputs)
        finally:
            self.template_backend = prev_backend

        fields = self._decision_fields(inputs)
        from swift.model.decision_head import QUESTION_TYPES
        encoded['decision_meta'] = {
            'n_options': [len(fields['options'])],
            'kinds': [QUESTION_TYPES[fields['kind']]],
        }
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        res = super()._post_encode(model, inputs)
        meta = inputs.get('decision_meta') if isinstance(inputs, dict) else None
        if meta is not None and isinstance(res, dict):
            res['decision_meta'] = meta
        return res


class JevTemplate(DecisionTemplate, Qwen3_5Template):
    _error_prefix = 'JEV'


register_template(QwenTemplateMeta(MLLMTemplateType.jev, template_cls=JevTemplate, default_system=None))


class GenericDecisionTemplate(DecisionTemplate, Template):

    _error_prefix = 'Decision'


register_template(
    TemplateMeta(
        LLMTemplateType.generic_decision,
        prefix=[],
        prompt=['{{QUERY}}'],
        chat_sep=None,
        template_cls=GenericDecisionTemplate,
        default_system=None))

# Clef's fixed system prompt / chat scaffolding
# The whole sequence is assembled by hand (NOT via a chat template) so the token spans the
# JointSchemaHead pools over land exactly where encode_record puts them.
_CLEF_SYSTEM_PROMPT = ('Read the complete state and schema. Decide every field jointly. Each answer '
                       'must be exactly one of that field\'s allowed options.')
_CLEF_PREFIX_TMPL = '<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSTATE:\n'
_CLEF_SUFFIX = '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:'
_CLEF_NOUL_CRITERIA = {
    'true': 'The proposition is true or the answer is yes.',
    'false': 'The proposition is false or the answer is no.',
}


class ClefTemplate(Qwen3_5Template):
    """Output token layout：
    [prefix: system prompt + "STATE:"]  [state]  [schema: SCHEMA FIELDS + FIELD
    block for each question]  [suffix: assistant opener]
    """

    @staticmethod
    def _clef_render(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        if getattr(inputs, 'images', None) or getattr(inputs, 'videos', None):
            raise NotImplementedError(
                'ClefTemplate supports text-only records this phase (Clef is a text decision model); '
                'images/videos require the deferred multimodal span-offset port.')
        from swift.model.decision_head import QUESTION_TYPES
        state, questions = self._clef_fields(inputs)

        schema_ids: List[int] = self._tokenize('\n\nSCHEMA FIELDS:\n')
        question_spans: List[Tuple[int, int]] = []
        option_spans: List[List[Tuple[int, int]]] = []
        kinds_int: List[int] = []
        n_options: List[int] = []
        for q_index, (qid, kind, instructions, criteria) in enumerate(questions):
            schema_ids += self._tokenize(f'\nFIELD {q_index + 1}\nID: {qid}\nTYPE: {kind}\nINSTRUCTION: ')
            question_start = len(schema_ids)
            if instructions is None or instructions == '':
                instructions = str(qid)
            schema_ids += self._tokenize(self._clef_render(instructions))
            question_end = len(schema_ids)
            schema_ids += self._tokenize('\nALLOWED OPTIONS:\n')
            spans: List[Tuple[int, int]] = []
            for o_index, (option_id, description) in enumerate(self._question_options(kind, criteria)):
                schema_ids += self._tokenize(f'OPTION {o_index + 1}: ')
                option_start = len(schema_ids)
                semantics: Dict[str, Any] = {'option_id': option_id}
                if description is not None:
                    semantics['description'] = description
                schema_ids += self._tokenize(self._clef_render(semantics))
                spans.append((option_start, len(schema_ids)))
                schema_ids += self._tokenize('\n')
            schema_ids += self._tokenize('END FIELD\n')
            question_spans.append((question_start, question_end))
            option_spans.append(spans)
            kinds_int.append(QUESTION_TYPES[kind])
            n_options.append(len(spans))

        prefix_ids = self._tokenize(_CLEF_PREFIX_TMPL.format(system=_CLEF_SYSTEM_PROMPT))
        suffix_ids = self._tokenize(_CLEF_SUFFIX)
        state_ids = self._tokenize(self._clef_render(state))
        if self.max_length is not None:
            fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
            if fixed_length > self.max_length:
                raise ValueError(
                    f'Clef schema requires {fixed_length} tokens before state; max_length={self.max_length}.')
            state_ids = state_ids[:self.max_length - fixed_length]
        schema_offset = len(prefix_ids) + len(state_ids)
        question_spans = [(s + schema_offset, e + schema_offset) for s, e in question_spans]
        option_spans = [[(s + schema_offset, e + schema_offset) for s, e in spans] for spans in option_spans]
        input_ids = prefix_ids + state_ids + schema_ids + suffix_ids
        if not input_ids or not questions:
            raise ValueError('Clef record produced no model input or questions.')
        return {
            'input_ids': input_ids,
            'decision_meta': {
                'n_options': n_options,
                'kinds': kinds_int,
                'question_spans': question_spans,
                'option_spans': option_spans,
            },
        }

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        meta = inputs.get('decision_meta') if isinstance(inputs, dict) else None
        if meta is not None and isinstance(inputs, dict) and inputs.get('input_ids') is not None:
            meta = dict(meta)
            meta['input_ids'] = inputs['input_ids']
            inputs = {**inputs, 'decision_meta': meta}
        res = super()._post_encode(model, inputs)
        if meta is not None and isinstance(res, dict):
            res['decision_meta'] = meta
        return res

    def _clef_fields(self, inputs: StdTemplateInputs) -> Tuple[Any, List[Tuple[str, str, Any, Any]]]:
        extra = inputs.extra_kwargs
        qids = extra.get('question_ids')
        kinds = extra.get('kinds')
        instructions = extra.get('questions')
        criteria = extra.get('criteria')
        if not qids or not kinds or instructions is None or criteria is None:
            raise ValueError(
                'ClefTemplate requires `question_ids` / `kinds` / `questions` / `criteria` in extra_kwargs '
                f'(produced by ClefPreprocessor); got qids={qids!r}, kinds={kinds!r}, '
                f'questions={instructions!r}, criteria={criteria!r}.')
        if not (len(qids) == len(kinds) == len(instructions) == len(criteria)):
            raise ValueError(f'Clef field lists disagree in length: qids={len(qids)}, kinds={len(kinds)}, '
                             f'instructions={len(instructions)}, criteria={len(criteria)}.')
        state = extra.get('state')
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except (json.JSONDecodeError, TypeError):
                pass
        if state is None:
            state = DecisionTemplate._get_state(inputs)
        criteria = [json.loads(c) if isinstance(c, str) else c for c in criteria]
        questions = [(str(qids[i]), DecisionTemplate._normalize_kind(kinds[i]), instructions[i], criteria[i])
                     for i in range(len(qids))]
        return state, questions

    @staticmethod
    def _question_options(kind: str, criteria: Any) -> List[Tuple[str, Any]]:
        from swift.model.decision_head import clef_option_ids
        ids = clef_option_ids(kind, criteria)
        if kind == 'noul':
            merged = dict(_CLEF_NOUL_CRITERIA)
            merged.update(criteria or {})
            return [(key, merged[key]) for key in ids]
        if kind == 'choice':
            crit = {str(k): v for k, v in (criteria or {}).items()}
            return [(key, crit[key]) for key in ids]
        # score
        crit = criteria or []
        return [(ids[i], crit[i]) for i in range(len(crit))]


register_template(QwenTemplateMeta(MLLMTemplateType.clef, template_cls=ClefTemplate, default_system=None))

# OmniJev option markers imported from the single definition site (omnijev_branch.py).
# Lazy import: the constants are resolved inside _encode to avoid a
# swift.template -> swift.model load cycle (swift.model.models.* import swift.template).
_OMNIJEV_THINK_PREFIX = '<think>\n'

_OMNIJEV_MAX_PIXELS = 768 * 28 * 28


class OmniJevTemplate(Qwen3_5Template):
    """OmniJev System-1 template (decision G1 branch + H media).

    OmniJev encodes ONE image per record and reads a per-option hidden state, so it reuses the
    standard media pipeline
    """

    @staticmethod
    def _omnijev_min_pixels(image_processor) -> int:
        size = getattr(image_processor, 'size', None)
        for key in ('shortest_edge', 'min_pixels'):
            val = None
            if size is not None:
                val = size.get(key) if hasattr(size, 'get') else getattr(size, key, None)
            if val is None:
                val = getattr(image_processor, key, None)
            if val:
                return int(val)
        return 65536

    @staticmethod
    def _find_subseq(seq: List[int], sub: List[int]) -> List[int]:
        n, m = len(seq), len(sub)
        if m == 0 or n < m:
            return []
        return [k for k in range(n - m + 1) if seq[k:k + m] == sub]

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            image_processor = self.processor.image_processor
            inputs.chat_template_kwargs.setdefault('min_pixels', self._omnijev_min_pixels(image_processor))
            inputs.chat_template_kwargs.setdefault('max_pixels', _OMNIJEV_MAX_PIXELS)
        return super().replace_tag(media_type, index, inputs)

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        self._omnijev_compose(inputs)
        super()._preprocess_inputs(inputs)

    def _omnijev_compose(self, inputs: StdTemplateInputs) -> None:
        from swift.model import omnijev_media as OM
        extra = inputs.extra_kwargs
        questions = extra.get('questions')
        options = extra.get('options')
        column_keys = extra.get('omnijev_column_keys')
        if not questions or not options or not column_keys:
            raise ValueError('OmniJevTemplate requires `questions` / `options` / `omnijev_column_keys` in extra_kwargs '
                             f'(produced by OmniJevPreprocessor); got questions={questions!r}, options={options!r}, '
                             f'omnijev_column_keys={column_keys!r}.')
        instruction = questions[0]
        opt_texts = [str(t) for t in options[0]]
        video = extra.get('omnijev_video')
        images = list(inputs.images or [])
        body = OM.option_body(opt_texts)
        if images:
            img = OM.compose_state(images, video)
            frames, content = OM.video_content(img, video, instruction)
            content_text = OM.content_to_text(content)
            inputs.images = frames
        else:
            content_text = instruction
            inputs.images = None
        inputs.messages = [
            {
                'role': 'user',
                'content': content_text
            },
            {
                'role': 'assistant',
                'content': _OMNIJEV_THINK_PREFIX + body
            },
        ]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from swift.model.omnijev_branch import OPT_CLOSE, OPT_OPEN
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        open_ids = self._tokenize(OPT_OPEN)
        close_ids = self._tokenize(OPT_CLOSE)
        if not open_ids or not close_ids:
            raise ValueError(f'OmniJev option markers tokenized empty (open={open_ids!r}, close={close_ids!r}); the '
                             'checkpoint tokenizer must have the two option tokens added (OmniJevModelLoader does this '
                             'via add_option_tokens before the template encodes).')
        opens = self._find_subseq(input_ids, open_ids)
        closes_last = [c + len(close_ids) - 1 for c in self._find_subseq(input_ids, close_ids)]
        if len(opens) != len(closes_last):
            raise ValueError(f'OmniJev marker mismatch: {len(opens)} open vs {len(closes_last)} close markers in the '
                             'encoded sequence (a stray marker inside an option text would desync the spans).')
        if not opens:
            raise ValueError('OmniJev encoded sequence has no option markers; the option body is empty.')
        markers = list(zip(opens, closes_last))
        extra = inputs.extra_kwargs
        column_keys = extra['omnijev_column_keys'][0]
        kind = DecisionTemplate._normalize_kind(extra['kinds'][0])
        from swift.model.decision_head import QUESTION_TYPES
        n_images = len(inputs.images or [])
        encoded['decision_meta'] = {
            'n_options': [len(column_keys)],
            'kinds': [QUESTION_TYPES[kind]],
            'option_markers': [markers],
            'mm_counts': {
                'image': n_images,
                'video': 0
            },
            'question_spans': [(opens[0] - 1, opens[0])],
            'option_spans': [markers],
        }
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def _extract_extra_decision_meta(self, meta: Optional[Dict[str, Any]], rec_i: int, flat: Dict[str, Any]) -> None:
        """Flatten OmniJev-specific meta fields into the collator's `flat` dict.

        `option_markers` is per-question (extended like `option_spans`);
        `mm_counts` is per-record (appended like `counts`).
        Without this override these fields would be silently dropped by the shared collator,
        causing a KeyError in `OmniJevModel.forward` / `OmniJevHead.forward`.
        """
        if meta is None:
            return
        flat.setdefault('option_markers', []).extend(meta.get('option_markers') or [])
        flat.setdefault('mm_counts', []).append(meta.get('mm_counts') or {'image': 0, 'video': 0})


register_template(QwenTemplateMeta(MLLMTemplateType.omnijev, template_cls=OmniJevTemplate, default_system=None))
