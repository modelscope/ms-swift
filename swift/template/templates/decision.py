# Copyright (c) ModelScope Contributors. All rights reserved.
"""Templates for the `decision` task_type (typed-decision System-1 scoring models).

Each decision model renders its own prompt layout and records the per-question locating meta the
scoring head consumes (`decision_meta`, see `swift/model/decision_head.py`). These templates do NOT
introduce a shared `ScoringTemplate` base (plan decision B): the decision encode/collate/decode
plumbing lives on the base `Template`, and each model subclass only overrides the rendering.
"""
import json
import string
from typing import Any, Dict, List, Tuple

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context
from .qwen import Qwen3_5Template, QwenTemplateMeta

# NOTE: `QUESTION_TYPES` lives in `swift.model.decision_head`, but a module-level import here would
# create a swift.template -> swift.model load cycle (swift.model.models.* import swift.template).
# Following base.py's `_decode_decision`, it is imported lazily inside the methods that need it.

# serve_decide.py's `S['labels']` are the single-token option letters A, B, ... (A..P for the native
# 16-slot choice head; JevVerbalizerHead rejects n > 16, so ascii_uppercase always covers them).
_OPTION_LETTERS = string.ascii_uppercase


class JevTemplate(Qwen3_5Template):
    """JEV System-1 template: render `serve_decide.py::s1_pass`'s BARE completion prompt.

    JEV-27B-VL's factory `config.json` is `Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`, so
    this subclasses `Qwen3_5Template` to reuse its tokenizer and Qwen3-VL media-token expansion. But
    JEV's System-1 pass is NOT a chat turn: s1_pass feeds vLLM a raw `CompletionRequest` (profile
    prefix='', `add_special_tokens=False`) with this exact layout (serve_decide.py line 252-253),
    each field separated by a single newline and the whole prompt ending at `[decision]:` (no
    trailing newline)::

        [kind] {kind}
        [state] {state}
        [question] {question}
        [options]
        {option lines}
        [decision]:

    where an option line is `f"{LETTER}) {opt}"` for `choice` (letters A..P, matching the trained
    verbalizer head) and the bare option text for `noul` / `score`. The readout is the next-token
    distribution at the final ':' token, which `JevVerbalizerHead` reads via the "last real token"
    rule (robust to left/right padding and to left-truncation of an over-long state).

    Two base behaviors are overridden to keep the prompt byte-faithful and the meta alive:
      * `_swift_prepare_inputs` skips `Qwen3_5Template`'s chat-only content `|trim` / `<think>`
        rewrapping -- s1_pass passes the state through verbatim.
      * `_encode` forces the swift backend (the bare prompt is built in `_swift_encode`; the jinja
        backend would wrap it in Qwen3.5 chat scaffolding) and attaches `decision_meta`.
      * `_post_encode` re-injects `decision_meta`, because `Qwen3_5Template._post_encode` ->
        `Qwen2VLTemplate._post_encode` returns a fresh `{'inputs_embeds': ...}` dict during training
        and the `pre_forward_hook` whitelist does not carry `decision_meta` (framework wiring note 8).

    After `JevPreprocessor`'s fan-out every record holds exactly ONE question, so `decision_meta`
    carries single-element lists (`counts` is filled by the collator as [1]).
    """

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        # Bypass Qwen3_5Template's chat-template content normalization; call the base merger directly.
        Template._swift_prepare_inputs(self, inputs)

    def _swift_encode(self, inputs: StdTemplateInputs):
        fields = self._decision_fields(inputs)
        content = self._render_s1_prompt(fields)
        # One OTHER context, all-zero loss scale: decision has no token-level labels (scored per
        # option), so nothing is a training target at the token level.
        return [content], [0.], 0

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # JEV's bare prompt is built by `_swift_encode`; never let a user `--template_backend jinja`
        # silently swap in the Qwen3.5 chat template.
        prev_backend = self.template_backend
        self.template_backend = 'swift'
        try:
            encoded = super()._encode(inputs)
        finally:
            self.template_backend = prev_backend

        fields = self._decision_fields(inputs)
        n_options = len(fields['options'])
        seq_len = len(encoded['input_ids'])
        from swift.model.decision_head import QUESTION_TYPES
        encoded['decision_meta'] = {
            'n_options': [n_options],
            'kinds': [QUESTION_TYPES[fields['kind']]],
            # The readout is the last real token; the head resolves its absolute column from the
            # collator's padding side / seq_lens, so this span is informational (kept for parity with
            # the span-pooling Clef/OmniJev heads).
            'question_spans': [(seq_len - 1, seq_len)],
            'option_spans': [[]],
        }
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        res = super()._post_encode(model, inputs)
        meta = inputs.get('decision_meta') if isinstance(inputs, dict) else None
        if meta is not None and isinstance(res, dict):
            res['decision_meta'] = meta
        return res

    def _decision_fields(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """Pull this record's single question (kind / question / options) from `extra_kwargs` and its
        state from the user message. `JevPreprocessor` emits exactly one question per record."""
        extra = inputs.extra_kwargs
        kinds = extra.get('kinds')
        questions = extra.get('questions')
        options = extra.get('options')
        if not kinds or not questions or not options:
            raise ValueError(
                'JevTemplate requires `kinds` / `questions` / `options` in extra_kwargs '
                '(produced by JevPreprocessor); got '
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
            raise ValueError(f'JEV question kind must be one of {list(QUESTION_TYPES)}, got {kind!r}.')
        return kind

    def _render_s1_prompt(self, fields: Dict[str, Any]) -> str:
        kind = fields['kind']
        options: List[str] = fields['options']
        if kind == 'choice':
            if len(options) > len(_OPTION_LETTERS):
                raise ValueError(f'JEV choice supports at most {len(_OPTION_LETTERS)} options, got {len(options)}.')
            lines = [f'{_OPTION_LETTERS[i]}) {opt}' for i, opt in enumerate(options)]
        else:
            lines = [str(opt) for opt in options]
        return (f'[kind] {kind}\n[state] {fields["state"]}\n[question] {fields["question"]}\n'
                '[options]\n' + '\n'.join(lines) + '\n[decision]:')


register_template(QwenTemplateMeta(MLLMTemplateType.jev, template_cls=JevTemplate, default_system=None))

# Clef's fixed system prompt / chat scaffolding, byte-copied from joint_schema_model.py (lines 24-27,
# 150-157). The whole sequence is assembled by hand (NOT via a chat template) so the token spans the
# JointSchemaHead pools over land exactly where encode_record puts them.
_CLEF_SYSTEM_PROMPT = (
    'Read the complete state and schema. Decide every field jointly. Each answer '
    'must be exactly one of that field\'s allowed options.')
_CLEF_PREFIX_TMPL = '<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSTATE:\n'
_CLEF_SUFFIX = '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:'
_CLEF_NOUL_CRITERIA = {
    'true': 'The proposition is true or the answer is yes.',
    'false': 'The proposition is false or the answer is no.',
}


def _clef_render(value: Any) -> str:
    """joint_schema_model.py::render -- a str passes through; anything else is compact sorted JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


class ClefTemplate(Qwen3_5Template):
    """Clef joint-schema template: render `joint_schema_model.py::encode_record`'s SCHEMA FIELDS layout.

    Clef's backbone is also `Qwen3_5ForConditionalGeneration` (config.json: model_type=qwen3_5,
    hidden 5120), so this subclasses `Qwen3_5Template` for the tokenizer / processor. Unlike JEV,
    Clef decodes ALL of a record's questions JOINTLY in ONE sequence, and its `JointSchemaHead` pools
    the hidden state over exact token spans, so `_encode` is fully overridden to replicate
    `encode_record` piece-wise: it tokenizes each schema fragment with `add_special_tokens=False`
    (== `self._tokenize`) and records, per question, the `question_span` (its instruction tokens) and
    per option the `option_span` (its rendered `{"option_id":..,"description":..}` tokens), then
    shifts every span by `schema_offset = len(prefix) + len(state)`.

    Two base behaviors are handled:
      * `_post_encode` stashes the padded `input_ids` into `decision_meta` before `Qwen3_5Template`
        -> `Qwen2VLTemplate._post_encode` converts them to `inputs_embeds` (training) and drops them;
        the head needs `input_ids` for its output-embedding lexical prior. It also re-injects the meta
        (the `pre_forward_hook` whitelist does not carry `decision_meta`).
      * Over-long states are truncated INSIDE `_encode` (state only, exactly like encode_record) so
        the spans stay valid; `_encode_truncated` then sees a sequence already within `max_length`.

    SCOPE (plan/comparison): Clef is a TEXT decision model (decision-models-comparison.md lists its
    modality as 文本). `encode_record` supports images/videos, but a faithful multimodal port must
    splice processor-expanded media tokens into the prefix and shift `schema_offset` accordingly, and
    its interaction with swift's training-time `inputs_embeds` merge is unverified -- so media is
    rejected here for now (deferred, surfaced to the user).
    """

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
            schema_ids += self._tokenize(_clef_render(instructions))
            question_end = len(schema_ids)
            schema_ids += self._tokenize('\nALLOWED OPTIONS:\n')
            spans: List[Tuple[int, int]] = []
            for o_index, (option_id, description) in enumerate(self._question_options(kind, criteria)):
                schema_ids += self._tokenize(f'OPTION {o_index + 1}: ')
                option_start = len(schema_ids)
                semantics: Dict[str, Any] = {'option_id': option_id}
                if description is not None:
                    semantics['description'] = description
                schema_ids += self._tokenize(_clef_render(semantics))
                spans.append((option_start, len(schema_ids)))
                schema_ids += self._tokenize('\n')
            schema_ids += self._tokenize('END FIELD\n')
            question_spans.append((question_start, question_end))
            option_spans.append(spans)
            kinds_int.append(QUESTION_TYPES[kind])
            n_options.append(len(spans))

        prefix_ids = self._tokenize(_CLEF_PREFIX_TMPL.format(system=_CLEF_SYSTEM_PROMPT))
        suffix_ids = self._tokenize(_CLEF_SUFFIX)
        state_ids = self._tokenize(_clef_render(state))
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
            # Stash the padded input_ids for the head's lexical prior; Qwen3_5's training-time
            # _post_encode returns a fresh {'inputs_embeds': ...} and drops input_ids.
            meta = dict(meta)
            meta['input_ids'] = inputs['input_ids']
            inputs = {**inputs, 'decision_meta': meta}
        res = super()._post_encode(model, inputs)
        if meta is not None and isinstance(res, dict):
            res['decision_meta'] = meta
        return res

    def _clef_fields(self, inputs: StdTemplateInputs) -> Tuple[Any, List[Tuple[str, str, Any, Any]]]:
        """Return `(state, [(question_id, kind, instructions, criteria), ...])` in record order.

        `ClefPreprocessor` emits the raw `state` (any JSON value) plus parallel per-question lists
        `question_ids` / `kinds` / `questions` (instructions) / `criteria`; the order is the record's
        original `questions` dict order, which the joint decode and the gold targets both rely on.
        """
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
            raise ValueError(
                f'Clef field lists disagree in length: qids={len(qids)}, kinds={len(kinds)}, '
                f'instructions={len(instructions)}, criteria={len(criteria)}.')
        state = extra.get('state')
        if state is None:
            state = self._get_state(inputs)
        questions = [(str(qids[i]), self._normalize_kind(kinds[i]), instructions[i], criteria[i])
                     for i in range(len(qids))]
        return state, questions

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
            raise ValueError(f'Clef question kind must be one of {list(QUESTION_TYPES)}, got {kind!r}.')
        return kind

    @staticmethod
    def _question_options(kind: str, criteria: Any) -> List[Tuple[str, Any]]:
        """joint_schema_model.py::question_options -- the option (id, description) list, in the exact
        order the head scores and the gold one-hot indexes: noul -> [('true',..),('false',..)],
        choice -> criteria items sorted by option id, score -> [(str(i), level) for i,level]."""
        if kind == 'noul':
            merged = dict(_CLEF_NOUL_CRITERIA)
            merged.update(criteria or {})
            return [(key, merged[key]) for key in ('true', 'false')]
        if kind == 'choice':
            return sorted((str(key), value) for key, value in (criteria or {}).items())
        return [(str(index), value) for index, value in enumerate(criteria or [])]


register_template(QwenTemplateMeta(MLLMTemplateType.clef, template_cls=ClefTemplate, default_system=None))


# OmniJev option markers (mso/records.py). The OmniJev checkpoint's chat_template.jinja emits, for
# add_generation_prompt with `enable_thinking` undefined, exactly the assistant opener followed by a
# single opening think tag + newline before the appended option body (verified against
# snapshots/.../chat_template.jinja lines 147-153). swift's chatml meta renders
# `user-opener {{QUERY}} user-closer assistant-opener` + assistant_content + suffix, so the faithful
# scaffold is reproduced by putting THINK_PREFIX + body in the assistant message and registering with a
# plain (non-thinking) QwenTemplateMeta: is_thinking=False keeps `_add_non_thinking_prefix` from
# injecting qwen3_5's non_thinking_prefix (a DIFFERENT closed think block that would move the prefix
# boundary L = opens[0]-1 off the token z_q is read at). The trailing suffix swift adds lands after the
# last option close marker, so it is in neither the branch prefix (ids[:L]) nor any option row.
_OMNIJEV_OPT_OPEN = '<|opt|>'
_OMNIJEV_OPT_CLOSE = '<|/opt|>'
_OMNIJEV_THINK_PREFIX = '<think>\n'

# OmniJev's official inference (`mso/infer.py::MSO1.__init__`) builds its processor with
# `AutoProcessor.from_pretrained(ckpt, max_pixels=768*28*28)` and lets the HF Qwen3.5-VL image processor
# resize each still with its OWN `smart_resize`: min_pixels = the processor's built-in `shortest_edge`
# (65536 = 256^2, which UPSCALES small stills) and max_pixels = 768*28*28 = 602112. swift's shared
# Qwen-VL path instead resizes with `qwen_vl_utils.fetch_image` at ITS defaults
# (IMAGE_MIN_TOKEN_NUM * factor^2 = 4 * 32^2 = 4096), which does NOT upscale: a 224px still yields a
# 14x14 grid (49 image tokens) instead of the checkpoint's 16x16 (64). Every option-marker position and
# the z_q readout token then shift off the values the head was trained on, silently corrupting the
# decision numerics with no error. `qwen_vl_utils.smart_resize` and the HF processor's `smart_resize` are
# the SAME algorithm at the SAME factor (patch_size * merge_size), so forcing the oracle's exact bounds
# through `fetch_image` reproduces its `image_grid_thw` byte-for-byte (verified across sizes). max_pixels
# is the oracle's fixed MSO1 default; min_pixels is read live from the processor so it tracks the
# checkpoint's preprocessor_config.json (the oracle likewise inherits it rather than passing it).
_OMNIJEV_MAX_PIXELS = 768 * 28 * 28


def _omnijev_min_pixels(image_processor) -> int:
    """The processor's built-in `smart_resize` floor (Qwen3.5-VL ships `shortest_edge` = 65536 = 256^2),
    which the oracle inherits by NOT passing `min_pixels`. Read it live (SizeDict exposes it by attribute
    or key) so swift upscales small stills exactly like the official inference path."""
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


def _find_subseq(seq: List[int], sub: List[int]) -> List[int]:
    """All start indices of `sub` in `seq` (verbatim `mso/infer.py::_spans_of` marker scan)."""
    n, m = len(seq), len(sub)
    if m == 0 or n < m:
        return []
    return [k for k in range(n - m + 1) if seq[k:k + m] == sub]


class OmniJevTemplate(Qwen3_5Template):
    """OmniJev System-1 template (decision G1 branch + H media).

    OmniJev's backbone is the hybrid Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`), so this subclasses
    `Qwen3_5Template` for the tokenizer / processor / Qwen3-VL media-token expansion. Unlike JEV (a bare
    text completion) and Clef (a hand-built text schema), OmniJev encodes ONE image per record and reads
    a per-option hidden state, so it reuses the standard media pipeline and only overrides three points:

      * `_preprocess_inputs` composes the official media BEFORE swift loads/rescales `inputs.images`:
        several stills -> one numbered panel, a video mosaic -> its 16 timestamped frames, a single still
        -> itself (`swift/model/omnijev_media.py`, a verbatim port of mso/panels.py + video.py +
        records.video_content). It then rebuilds the messages as [user(media-interleaved content),
        assistant(think-prefix + option body)] so the rendered sequence byte-matches official
        `apply_chat_template([user], add_generation_prompt=True) + body`.
      * `_encode` reuses `Qwen3VLTemplate._encode` (replace_tag -> image_processor -> _extend_tokens, the
        same expansion official's AutoProcessor does), then locates the option markers in the EXPANDED
        input_ids (the space the branch forward indexes) and attaches `decision_meta` with, per question,
        the marker positions `[(open, close_last), ...]` and, per record, the media counts the loader uses
        to slice the collator-concatenated pixel tensors back per record.
      * `_post_encode` returns `inputs` unchanged: the branch forward re-runs the backbone per option row,
        so it needs input_ids + pixel_values + image_grid_thw (Qwen3_5's training-time conversion to
        inputs_embeds would drop them), and the pre_forward_hook whitelist carries neither pixels nor
        decision_meta, so whatever _post_encode returns must already contain them.

    After `OmniJevPreprocessor`'s fan-out every record holds exactly ONE question, so `decision_meta`
    carries single-element per-question lists (`counts` is filled by the collator as [1]).
    """

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        # Force the oracle's image-resize bounds (see `_OMNIJEV_MAX_PIXELS` / `_omnijev_min_pixels`).
        # `Qwen2VLTemplate.replace_tag` already merges `inputs.chat_template_kwargs` into the
        # `qwen_vl_utils.fetch_image` element dict, so injecting min_pixels/max_pixels there reuses the
        # parent's EXACT resize + vision-token insertion (the config's vision_start/end ids) and only
        # changes the smart_resize bounds. Returning the tokens ourselves would re-tokenize the ambiguous
        # 'lparr' literal to a duplicate id (248060 vs the config's 248053) and desync the sequence.
        if media_type == 'image':
            image_processor = self.processor.image_processor
            inputs.chat_template_kwargs.setdefault('min_pixels', _omnijev_min_pixels(image_processor))
            inputs.chat_template_kwargs.setdefault('max_pixels', _OMNIJEV_MAX_PIXELS)
        return super().replace_tag(media_type, index, inputs)

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        # Compose the official media + rebuild the messages BEFORE super() loads/rescales inputs.images,
        # so the composed pixels match the checkpoint's training input (official composes on the original
        # files, then hands them to AutoProcessor(max_pixels=768*28*28)).
        self._omnijev_compose(inputs)
        super()._preprocess_inputs(inputs)

    def _omnijev_compose(self, inputs: StdTemplateInputs) -> None:
        from swift.model import omnijev_media as OM
        extra = inputs.extra_kwargs
        questions = extra.get('questions')
        options = extra.get('options')
        column_keys = extra.get('omnijev_column_keys')
        if not questions or not options or not column_keys:
            raise ValueError(
                'OmniJevTemplate requires `questions` / `options` / `omnijev_column_keys` in extra_kwargs '
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
            {'role': 'user', 'content': content_text},
            {'role': 'assistant', 'content': _OMNIJEV_THINK_PREFIX + body},
        ]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        open_ids = self._tokenize(_OMNIJEV_OPT_OPEN)
        close_ids = self._tokenize(_OMNIJEV_OPT_CLOSE)
        if not open_ids or not close_ids:
            raise ValueError(
                f'OmniJev option markers tokenized empty (open={open_ids!r}, close={close_ids!r}); the '
                'checkpoint tokenizer must have the two option tokens added (OmniJevModelLoader does this '
                'via add_option_tokens before the template encodes).')
        opens = _find_subseq(input_ids, open_ids)
        closes_last = [c + len(close_ids) - 1 for c in _find_subseq(input_ids, close_ids)]
        if len(opens) != len(closes_last):
            raise ValueError(
                f'OmniJev marker mismatch: {len(opens)} open vs {len(closes_last)} close markers in the '
                'encoded sequence (a stray marker inside an option text would desync the spans).')
        if not opens:
            raise ValueError('OmniJev encoded sequence has no option markers; the option body is empty.')
        markers = list(zip(opens, closes_last))
        extra = inputs.extra_kwargs
        column_keys = extra['omnijev_column_keys'][0]
        kind = self._normalize_kind(extra['kinds'][0])
        from swift.model.decision_head import QUESTION_TYPES
        n_images = len(inputs.images or [])
        encoded['decision_meta'] = {
            # column width the head emits (noul=2 / choice=K+1 / score=L), NOT the option-block count.
            'n_options': [len(column_keys)],
            'kinds': [QUESTION_TYPES[kind]],
            # per question (fan-out => one per record): the [(open, close_last), ...] marker positions in
            # the FULL media-expanded sequence; the branch derives L = opens[0]-1 and slices option rows.
            'option_markers': [markers],
            # per RECORD media counts, so the loader slices the collator-concatenated pixel_values /
            # image_grid_thw back to this record for the prefix pass.
            'mm_counts': {'image': n_images, 'video': 0},
            'question_spans': [(opens[0] - 1, opens[0])],  # informational: the z_q readout token
            'option_spans': [markers],
        }
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # The branch forward (decision G1) re-runs the backbone per option row, so it needs input_ids +
        # pixel_values + image_grid_thw; do NOT convert to inputs_embeds (Qwen3_5's training-time
        # _post_encode drops them). Return inputs unchanged -- like Qwen3VLTemplate._post_encode -- so the
        # pixels AND decision_meta survive: the pre_forward_hook whitelist carries neither, and whatever
        # _post_encode returns REPLACES the forward kwargs.
        return inputs

    @staticmethod
    def _normalize_kind(kind: Any) -> str:
        from swift.model.decision_head import QUESTION_TYPES
        kind = str(kind).strip().lower()
        if kind not in QUESTION_TYPES:
            raise ValueError(f'OmniJev question kind must be one of {list(QUESTION_TYPES)}, got {kind!r}.')
        return kind


register_template(QwenTemplateMeta(MLLMTemplateType.omnijev, template_cls=OmniJevTemplate, default_system=None))


class GenericDecisionTemplate(Template):
    """Generic decision template for training any CausalLM into a JEV-style decision model.

    Unlike ``JevTemplate`` (which subclasses ``Qwen3_5Template`` for Qwen3-VL media-token expansion
    and chat-template normalization), this subclasses the base ``Template`` directly, so it works
    with any text CausalLM (Llama, Qwen2.5, Mistral, etc.) without Qwen3.5-specific assumptions.

    Renders the same bare completion prompt as ``JevTemplate``::

        [kind] {kind}
        [state] {state}
        [question] {question}
        [options]
        {option lines}
        [decision]:

    where an option line is ``f"{LETTER}) {opt}"`` for ``choice`` (letters A..P, matching the
    verbalizer head) and the bare option text for ``noul`` / ``score``. The readout token is the last
    real token (the ':' of '[decision]:'), which ``JevVerbalizerHead`` reads via the "last real token"
    rule. After ``JevPreprocessor``'s fan-out every record holds exactly one question, so
    ``decision_meta`` carries single-element lists.

    This template is text-only (no media handling). For multimodal decision training use ``JevTemplate``
    with a Qwen3.5-VL base.
    """

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        # Bypass any chat-template content normalization; the bare prompt is built in _swift_encode.
        Template._swift_prepare_inputs(self, inputs)

    def _swift_encode(self, inputs: StdTemplateInputs):
        fields = self._decision_fields(inputs)
        content = self._render_s1_prompt(fields)
        # One OTHER context, all-zero loss scale: decision has no token-level labels (scored per
        # option), so nothing is a training target at the token level.
        return [content], [0.], 0

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # Force the swift backend (the bare prompt is built in _swift_encode; never let a
        # user --template_backend jinja swap in a chat template).
        prev_backend = self.template_backend
        self.template_backend = 'swift'
        try:
            encoded = super()._encode(inputs)
        finally:
            self.template_backend = prev_backend

        fields = self._decision_fields(inputs)
        n_options = len(fields['options'])
        seq_len = len(encoded['input_ids'])
        from swift.model.decision_head import QUESTION_TYPES
        encoded['decision_meta'] = {
            'n_options': [n_options],
            'kinds': [QUESTION_TYPES[fields['kind']]],
            'question_spans': [(seq_len - 1, seq_len)],
            'option_spans': [[]],
        }
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        res = super()._post_encode(model, inputs)
        meta = inputs.get('decision_meta') if isinstance(inputs, dict) else None
        if meta is not None and isinstance(res, dict):
            res['decision_meta'] = meta
        return res

    def _decision_fields(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """Pull this record's single question (kind / question / options) from `extra_kwargs` and its
        state from the user message. `JevPreprocessor` emits exactly one question per record."""
        extra = inputs.extra_kwargs
        kinds = extra.get('kinds')
        questions = extra.get('questions')
        options = extra.get('options')
        if not kinds or not questions or not options:
            raise ValueError(
                'GenericDecisionTemplate requires `kinds` / `questions` / `options` in extra_kwargs '
                '(produced by JevPreprocessor); got '
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
            raise ValueError(f'Decision question kind must be one of {list(QUESTION_TYPES)}, got {kind!r}.')
        return kind

    def _render_s1_prompt(self, fields: Dict[str, Any]) -> str:
        kind = fields['kind']
        options: List[str] = fields['options']
        if kind == 'choice':
            if len(options) > len(_OPTION_LETTERS):
                raise ValueError(f'Decision choice supports at most {len(_OPTION_LETTERS)} options, got {len(options)}.')
            lines = [f'{_OPTION_LETTERS[i]}) {opt}' for i, opt in enumerate(options)]
        else:
            lines = [str(opt) for opt in options]
        return (f'[kind] {kind}\n[state] {fields["state"]}\n[question] {fields["question"]}\n'
                '[options]\n' + '\n'.join(lines) + '\n[decision]:')


register_template(TemplateMeta(LLMTemplateType.generic_decision,
                               prefix=[], prompt=['{{QUERY}}'], chat_sep=None,
                               template_cls=GenericDecisionTemplate, default_system=None))
