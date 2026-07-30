# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import inspect
import torch
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from swift.template.base import Template as LegacyTemplate
from swift.template.template_inputs import StdTemplateInputs
from swift.utils import get_logger

logger = get_logger()


class NextTokenShiftMixin:
    """dev's entire delta over a legacy Template: labels become next-token at encode time.

    Kept separate from `Template` because build_template mixes it into the REAL legacy class
    (`shifted_template_class`) on the default path, while dev's own encode rewrite inherits it here --
    one definition of the label convention, two ways in. Must precede the legacy class in the MRO.

    Label convention: swift's legacy Template emits labels ALIGNED with input_ids
    (labels[i] == input_ids[i]) and shifts at loss time (HF convention). twinkle's forward computes
    logps via no-shift selective_log_softmax and therefore expects NEXT-token labels, which twinkle's
    own Template produces at encode time (_roll_labels). To stay consistent with twinkle ("whoever
    encodes, shifts") the shift happens here -- the InputProcessor must NOT touch label semantics.
    """

    # Records that a sample is already shifted, so a second encode cannot shift twice.
    SHIFTED_KEY = '_labels_shifted'

    @staticmethod
    def _shift_labels_next_token(labels: List[int]) -> List[int]:
        """Explicit next-token shift: out[i]=labels[i+1], out[-1]=-100 (NOT circular roll)."""
        if not labels:
            return labels
        return list(labels[1:]) + [-100]

    def encode(self, inputs, return_template_inputs: bool = False, return_length: bool = False):
        """Encode, then shift labels to next-token alignment (training mode only).

        vLLM-mode guard: the next-token shift must fire ONLY in training modes. In
        inference/rollout modes (vllm/lmdeploy/sglang/transformers) legacy `is_training` is False and
        `_encode` clears labels to None, so today the `labels is not None` check already skips the
        shift. We ALSO gate on `self.is_training` explicitly so a rollout input can never be silently
        shifted even if a future mode were to emit labels.
        """
        encoded = super().encode(inputs, return_template_inputs=return_template_inputs, return_length=return_length)
        if (self.is_training and isinstance(encoded, dict) and encoded.get('labels') is not None
                and not encoded.get(self.SHIFTED_KEY)):
            encoded['labels'] = self._shift_labels_next_token(list(encoded['labels']))
            if encoded.get('loss_scale') is not None:
                encoded['loss_scale'] = list(encoded['loss_scale'][1:]) + [0.0]
            encoded[self.SHIFTED_KEY] = True
        return encoded


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
        cls = type(name, (NextTokenShiftMixin, base), {'__module__': __name__, '__qualname__': name})
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


class Template(NextTokenShiftMixin, LegacyTemplate):
    # Override: standard media placeholder -> model HF placeholder that the processor later expands.
    #   e.g. {'<image>': '<|vision_start|><|image_pad|><|vision_end|>'}
    # Consumed by the default `replace_tag`; subclasses with version-dependent placeholders
    # (e.g. Qwen video) override `replace_tag` instead. The chat format itself is NOT declared here --
    # it is derived entirely from the legacy `TemplateMeta` (prefix/prompt/chat_sep/suffix/system_prefix).
    _media_tag_map: Dict[str, str] = {}

    @classmethod
    def from_template(cls, template: LegacyTemplate) -> 'Template':
        """Create a Template from an existing template instance."""
        new_template = copy.copy(template)
        new_template.__class__ = cls
        return new_template

    # ------------------------------------------------------------------
    # Subclass override points
    # ------------------------------------------------------------------

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs) -> List[str]:
        """Return the model HF placeholder for a media tag (no expansion; the processor expands it).

        Default reads `_media_tag_map`. Media *preprocessing* (fetch/resize) lives in `_prepare_media_inputs`,
        so unlike the legacy replace_tag this override has no side effects.
        """
        tag = f'<{media_type}>'
        return [self._media_tag_map.get(tag, tag)]

    def _build_text(self, inputs: StdTemplateInputs) -> str:
        """Build the full chat text, derived entirely from `template_meta` via legacy `_swift_encode`.

        Reuses the legacy assembly (prefix/bos/prompt/chat_sep/suffix/system/thinking/agent + `replace_tag`)
        and joins the resulting context list into one string (token-id segments such as bos/eos are decoded).
        The HF processor then tokenizes the whole string once and expands media placeholders. There is no
        per-subclass format declaration -- the format lives only in `TemplateMeta`.

        Inference vs training is implicit: `_swift_encode` omits the final suffix when is_training=False,
        so a trailing user turn renders through the assistant header for generation.
        """
        inp = copy.copy(inputs)
        inp.messages = deepcopy(inputs.messages)
        inp.image_idx = inp.video_idx = inp.audio_idx = 0
        res_context_list, loss_scale_list, _ = self._swift_encode(inp)
        res_context_list, _ = self._simplify_context_list(res_context_list, loss_scale_list, inp)
        parts = []
        for c in res_context_list:
            parts.append(c if isinstance(c, str) else self.tokenizer.decode(c))
        return ''.join(parts)

    def collate_mm_data(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Batch-level MM collation hook. Override for model-specific collation.
        Return None to use InputProcessor default VLM_CONCAT_FIELDS logic."""
        return None

    def post_collate(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Post-collation hook. Default: identity."""
        return batch

    def _postprocess_encoded(self, encoded: Dict[str, Any], inputs: StdTemplateInputs, input_ids: List[int]) -> None:
        """Post-encode hook: subclasses may add model-specific fields (e.g. token_type_ids) in place."""
        return

    def _prepare_media_inputs(self, inputs: StdTemplateInputs, hf_processor) -> dict:
        """Prepare media-related processor kwargs. Subclasses may override."""
        kwargs = {}
        images = inputs.images if inputs.images else None
        videos = inputs.videos if inputs.videos else None
        audios = getattr(inputs, 'audios', None) or None
        if images is not None:
            kwargs['images'] = images
        if videos is not None:
            kwargs['videos'] = videos
        if audios is not None:
            kwargs['audio'] = audios
        return kwargs

    # ------------------------------------------------------------------
    # TokenizeByRound: per-round encoding to locate assistant boundaries
    # ------------------------------------------------------------------

    def _build_text_partial(self,
                            inputs: StdTemplateInputs,
                            msg_count: int,
                            with_generation_prompt: bool = False) -> str:
        """Build text for the first msg_count messages by slicing and reusing meta-driven _build_text.

        `with_generation_prompt` is accepted for call-site compatibility but ignored: when the slice ends
        with a user turn, `_swift_encode` already renders through the assistant header (generation prompt).
        """
        original_messages = inputs.messages
        inputs.messages = original_messages[:msg_count]
        try:
            text = self._build_text(inputs)
        finally:
            inputs.messages = original_messages
        return text

    @staticmethod
    def _count_media_in_messages(messages: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """Count the media items in a message list.

        Returns:
            (n_images, n_videos, n_audios)
        """
        n_images = 0
        n_videos = 0
        n_audios = 0
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                n_images += content.count('<image>')
                n_videos += content.count('<video>')
                n_audios += content.count('<audio>')
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get('type', '')
                        if ptype == 'image':
                            n_images += 1
                        elif ptype == 'video':
                            n_videos += 1
                        elif ptype == 'audio':
                            n_audios += 1
        return n_images, n_videos, n_audios

    def _get_partial_media_kwargs(self, inputs: StdTemplateInputs, partial_messages: List[Dict[str, Any]],
                                  full_media_kwargs: dict) -> dict:
        """Build correct media kwargs for a subset of messages (pass only the matching count of images/videos)."""
        n_images, n_videos, n_audios = self._count_media_in_messages(partial_messages)
        kwargs = {}
        if 'images' in full_media_kwargs and n_images > 0:
            kwargs['images'] = full_media_kwargs['images'][:n_images]
        if 'videos' in full_media_kwargs and n_videos > 0:
            kwargs['videos'] = full_media_kwargs['videos'][:n_videos]
        if 'audio' in full_media_kwargs and n_audios > 0:
            kwargs['audio'] = full_media_kwargs['audio'][:n_audios]
        # Pass through non-media params such as do_resize.
        for k, v in full_media_kwargs.items():
            if k not in ('images', 'videos', 'audio'):
                kwargs[k] = v
        return kwargs

    def _build_labels_tokenize_by_round(self, input_ids: List[int], inputs: StdTemplateInputs, hf_processor,
                                        processor_kwargs: dict) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Locate assistant boundaries precisely via TokenizeByRound.

        Core logic:
        - For each assistant message, encode messages[:i] + generation_prompt to get start_pos.
        - Encode messages[:i+1] to get end_pos.
        - [start_pos, end_pos) is the assistant content token range.

        Args:
            input_ids: fully encoded token ids
            inputs: standard inputs
            hf_processor: HF processor instance
            processor_kwargs: processor kwargs used for the full encode (includes media)

        Returns:
            (labels, assistant_ranges): labels list and list of assistant ranges
        """
        labels = [-100] * len(input_ids)
        messages = inputs.messages
        assistant_ranges: List[Tuple[int, int]] = []

        # Extract media kwargs (drop return_tensors to get lists).
        media_kwargs = {k: v for k, v in processor_kwargs.items() if k != 'return_tensors'}

        for i, msg in enumerate(messages):
            if msg['role'] != 'assistant':
                continue

            # Encode messages[:i] + generation_prompt -> start_pos
            partial_text = self._build_text_partial(inputs, i, with_generation_prompt=True)
            partial_media = self._get_partial_media_kwargs(inputs, messages[:i], media_kwargs)
            partial_kwargs = {'return_tensors': 'pt'}
            partial_kwargs.update(partial_media)
            partial_encoded = hf_processor(text=[partial_text], **partial_kwargs)
            start_ids = partial_encoded['input_ids'][0].tolist()
            start_pos = len(start_ids)

            # The prefix (up to and including the assistant generation prompt) must be a genuine prefix
            # of the full encoding; otherwise the length-diff boundary is meaningless. Fail-fast here.
            self._assert_prefix_append_only(input_ids, start_ids, i)

            # Encode messages[:i+1] -> end_pos
            partial_text = self._build_text_partial(inputs, i + 1, with_generation_prompt=False)
            partial_media = self._get_partial_media_kwargs(inputs, messages[:i + 1], media_kwargs)
            partial_kwargs = {'return_tensors': 'pt'}
            partial_kwargs.update(partial_media)
            partial_encoded = hf_processor(text=[partial_text], **partial_kwargs)
            end_pos = len(partial_encoded['input_ids'][0])

            # Clamp to actual sequence length
            start_pos = min(start_pos, len(input_ids))
            end_pos = min(end_pos, len(input_ids))

            if start_pos < end_pos:
                assistant_ranges.append((start_pos, end_pos))
                labels[start_pos:end_pos] = input_ids[start_pos:end_pos]

        return labels, assistant_ranges

    @staticmethod
    def _assert_prefix_append_only(input_ids: List[int], prefix_ids: List[int], msg_index: int) -> None:
        """Verify that `prefix_ids` (a shorter-prefix encoding) is a true prefix of `input_ids`.

        TokenizeByRound locates assistant token ranges purely from encoding lengths, which is only
        correct when the chat template is append-only: a longer message prefix must extend the token
        sequence of a shorter one rather than rewrite earlier tokens. Templates that fold adjacent
        turns or strip prior thinking content violate this, and silently yield mislabeled data.
        Raise here so such templates are caught instead of training on wrong labels.
        """
        # A prefix longer than the full encoding can only mean the tail was truncated (handled by the
        # caller's clamp); the overlapping region is what must match.
        compare_len = min(len(prefix_ids), len(input_ids))
        if prefix_ids[:compare_len] == input_ids[:compare_len]:
            return
        mismatch = next((j for j in range(compare_len) if prefix_ids[j] != input_ids[j]), compare_len)
        raise ValueError(f'TokenizeByRound requires an append-only chat template, but the encoding of the message '
                         f'prefix (up to assistant message index {msg_index}) diverges from the full encoding at '
                         f'token position {mismatch}. This template likely rewrites earlier tokens (e.g. folds '
                         f'adjacent turns or clears prior thinking), so length-diff assistant boundaries would be '
                         f'wrong. A template-specific labels override is required for this model.')

    def _compute_loss_scale(self, input_ids: List[int], labels: List[int], assistant_ranges: List[Tuple[int, int]],
                            inputs: StdTemplateInputs) -> Optional[List[float]]:
        """Compute loss_scale, supporting default/last_round/thinking down-weighting.

        Args:
            input_ids: full token ids
            labels: labels produced by TokenizeByRound
            assistant_ranges: list of assistant token ranges [(start, end), ...]
            inputs: standard inputs

        Returns:
            loss_scale list (non-binary), or None (binary, labels=-100 is enough)
        """
        loss_scale_obj = self.loss_scale
        base_strategy = loss_scale_obj.base_strategy

        # Determine which assistant ranges participate in training based on base_strategy.
        if base_strategy == 'last_round' and assistant_ranges:
            active_ranges = [assistant_ranges[-1]]
        elif base_strategy == 'all':
            # 'all': every token participates (not just assistant).
            active_ranges = [(0, len(input_ids))]
        else:
            # 'default': all assistant ranges.
            active_ranges = assistant_ranges

        # last_round: regenerate labels to keep only the last turn.
        if base_strategy == 'last_round' and assistant_ranges:
            labels_new = [-100] * len(input_ids)
            start, end = assistant_ranges[-1]
            labels_new[start:end] = input_ids[start:end]
            labels[:] = labels_new
        elif base_strategy == 'all':
            # Train on all tokens.
            for i in range(len(labels)):
                labels[i] = input_ids[i]

        # Check whether a non-binary loss_scale is needed.
        is_binary = self.is_binary_loss_scale
        if is_binary is None:
            is_binary = loss_scale_obj.is_binary_loss_scale

        # Shortcut: a pure base LossScale that is binary can return None directly.
        from swift.loss_scale.base import LossScale as BaseLossScale
        is_pure_base = type(loss_scale_obj) is BaseLossScale

        if is_pure_base and is_binary:
            return None

        # Non-binary or custom loss_scale logic: build the loss_scale list.
        loss_scale = [0.0] * len(input_ids)

        # Get tokenizer for decoding.
        tokenizer = self.tokenizer

        for start, end in active_ranges:
            if start >= end:
                continue
            # Decode this range of tokens to text.
            segment_ids = input_ids[start:end]
            segment_text = tokenizer.decode(segment_ids, skip_special_tokens=False)

            # Call loss_scale_obj.get_loss_scale for fine-grained weighting.
            contexts, weights = loss_scale_obj.get_loss_scale(segment_text)

            # Map contexts + weights back to token level.
            token_offset = 0
            for ctx, weight in zip(contexts, weights):
                if isinstance(ctx, str):
                    ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
                    ctx_len = len(ctx_ids)
                else:
                    ctx_len = len(ctx) if isinstance(ctx, list) else 1

                # Assign the weight to the corresponding tokens.
                for j in range(ctx_len):
                    pos = start + token_offset + j
                    if pos < end and pos < len(loss_scale):
                        loss_scale[pos] = weight
                        if weight == 0.0:
                            labels[pos] = -100
                token_offset += ctx_len

            # If mapping is incomplete (decode/encode not perfectly symmetric), fill the rest with 1.0.
            for j in range(token_offset, end - start):
                pos = start + j
                if pos < len(loss_scale) and loss_scale[pos] == 0.0 and labels[pos] != -100:
                    loss_scale[pos] = 1.0

        if is_binary:
            # Binary mode: no loss_scale list needed, labels=-100 is enough.
            return None

        return loss_scale

    # ------------------------------------------------------------------
    # return_assistant_tokens_mask detection and template patching
    # ------------------------------------------------------------------

    @property
    def supports_assistant_tokens_mask(self) -> bool:
        """Detect whether processor.apply_chat_template supports return_assistant_tokens_mask.

        Conditions:
        1. processor has an apply_chat_template method
        2. chat_template contains {% generation %} markers
        3. the call returns a valid assistant_masks (0 < sum < len)
        """
        if hasattr(self, '_supports_assistant_mask'):
            return self._supports_assistant_mask

        hf_processor = self.processor
        self._supports_assistant_mask = False

        if not hasattr(hf_processor, 'apply_chat_template'):
            return False

        # Try to patch chat_template to add {% generation %} markers.
        self._patch_chat_template_for_generation(hf_processor)

        # Test with a simple dialogue.
        try:
            is_vlm = self.model_meta.is_multimodal
            if is_vlm:
                dummy_messages = [
                    {
                        'role': 'user',
                        'content': [{
                            'type': 'text',
                            'text': 'Hi'
                        }]
                    },
                    {
                        'role': 'assistant',
                        'content': [{
                            'type': 'text',
                            'text': 'Hello'
                        }]
                    },
                ]
            else:
                dummy_messages = [
                    {
                        'role': 'user',
                        'content': 'Hi'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Hello'
                    },
                ]
            result = hf_processor.apply_chat_template(
                dummy_messages, return_assistant_tokens_mask=True, return_dict=True, tokenize=True)
            if hasattr(result, 'keys') and 'assistant_masks' in result:
                # apply_chat_template (no return_tensors) returns nested lists for a single conversation.
                mask = result['assistant_masks'][0]
                total = len(mask)
                ones = sum(mask)
                self._supports_assistant_mask = (0 < ones < total)
                if self._supports_assistant_mask:
                    logger.info('Template: return_assistant_tokens_mask is supported.')
                else:
                    logger.debug(f'Template: assistant_masks invalid '
                                 f'(ones={ones}, total={total}), falling back to TokenizeByRound.')
        except Exception as e:
            logger.debug(f'Template: return_assistant_tokens_mask test failed: {e}')

        return self._supports_assistant_mask

    def _patch_chat_template_for_generation(self, hf_processor) -> None:
        """Check whether the template can enable {% generation %} markers.

        Note: Jinja2's {% generation %} marker cannot span if/else boundaries, so it cannot be
        auto-injected into complex templates (e.g. Qwen3.5-VL). This method only checks whether
        the template already natively contains {% generation %} markers. It will take effect
        automatically once upstream models support it natively.
        """
        # Detection only, no modification -- waiting for upstream templates to support {% generation %} natively.
        tokenizer = hf_processor.tokenizer if hasattr(hf_processor, 'tokenizer') else hf_processor
        tmpl = getattr(hf_processor, 'chat_template', None) or getattr(tokenizer, 'chat_template', None)
        if tmpl and ('{% generation %}' in tmpl or '{%generation%}' in tmpl):
            logger.debug('Template: chat_template already contains {% generation %} markers.')
        else:
            logger.debug('Template: chat_template does not contain {% generation %} markers. '
                         'return_assistant_tokens_mask will likely return all-zero mask.')

    # ------------------------------------------------------------------
    # apply_chat_template encoding path
    # ------------------------------------------------------------------

    def _build_messages_for_chat_template(self, inputs: StdTemplateInputs) -> List[Dict[str, Any]]:
        """Rebuild the messages format required by apply_chat_template from StdTemplateInputs.

        Converts '<image>'/'<video>'/'<audio>' placeholders back into list-of-dicts format.
        """
        messages = []
        system = self._get_system(inputs)
        if system:
            messages.append({'role': 'system', 'content': system})

        for msg in inputs.messages:
            role = msg['role']
            content = msg['content']

            if not isinstance(content, str):
                # Already list format or another type; use as-is.
                messages.append({'role': role, 'content': content})
                continue

            # Convert <image>/<video>/<audio> in the string to list-of-dicts.
            has_media = ('<image>' in content or '<video>' in content or '<audio>' in content)
            if not has_media:
                # Plain text - VLMs need list format.
                if self.model_meta.is_multimodal:
                    messages.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
                else:
                    messages.append({'role': role, 'content': content})
                continue

            # Has media placeholders - parse into list format.
            parts = []
            remaining = content
            while remaining:
                # Find the nearest placeholder.
                positions = []
                for tag, media_type in [('<image>', 'image'), ('<video>', 'video'), ('<audio>', 'audio')]:
                    pos = remaining.find(tag)
                    if pos >= 0:
                        positions.append((pos, tag, media_type))
                if not positions:
                    if remaining:
                        parts.append({'type': 'text', 'text': remaining})
                    break
                positions.sort()
                pos, tag, media_type = positions[0]
                if pos > 0:
                    parts.append({'type': 'text', 'text': remaining[:pos]})
                parts.append({'type': media_type})
                remaining = remaining[pos + len(tag):]

            if not parts:
                parts = [{'type': 'text', 'text': ''}]
            messages.append({'role': role, 'content': parts})

        return messages

    def _encode_via_chat_template(self, inputs: StdTemplateInputs, hf_processor, tokenizer) -> Optional[Dict[str, Any]]:
        """Encode via apply_chat_template + return_assistant_tokens_mask.

        Returns the full encoded dict on success, or None on failure (triggers fallback).
        """
        try:
            messages = self._build_messages_for_chat_template(inputs)

            # Build apply_chat_template kwargs.
            chat_kwargs = {
                'return_assistant_tokens_mask': True,
                'tokenize': True,
                'return_dict': True,
                'return_tensors': 'pt',
            }

            # Media kwargs.
            media_kwargs = self._prepare_media_inputs(inputs, hf_processor)

            # Check whether apply_chat_template supports processor_kwargs.
            sig = inspect.signature(hf_processor.apply_chat_template)
            supported_params = set(sig.parameters.keys())

            if 'processor_kwargs' in supported_params:
                # New interface: media kwargs go through processor_kwargs.
                processor_kwargs = {}
                if inputs.mm_processor_kwargs:
                    processor_kwargs.update(inputs.mm_processor_kwargs)
                # Media kwargs may contain do_resize etc.; put them into processor_kwargs.
                for k, v in list(media_kwargs.items()):
                    if k in supported_params:
                        chat_kwargs[k] = v
                    else:
                        processor_kwargs[k] = v
                if processor_kwargs:
                    chat_kwargs['processor_kwargs'] = processor_kwargs
            else:
                # Old interface: pass all kwargs directly.
                chat_kwargs.update(media_kwargs)
                if inputs.mm_processor_kwargs:
                    chat_kwargs.update(inputs.mm_processor_kwargs)

            result = hf_processor.apply_chat_template(messages, **chat_kwargs)

            if not hasattr(result, 'keys') or 'assistant_masks' not in result:
                return None

            # Extract input_ids and assistant_masks.
            # return_tensors='pt' -> tensors of shape [1, L]; take the single sample and flatten.
            input_ids = result['input_ids'][0].tolist()
            assistant_masks = result['assistant_masks'][0].tolist()

            # Verify alignment.
            if len(assistant_masks) != len(input_ids):
                logger.warning_once(f'assistant_masks length ({len(assistant_masks)}) != '
                                    f'input_ids length ({len(input_ids)}), falling back to TokenizeByRound.')
                return None

            # Verify the mask is valid (not all zeros).
            if sum(assistant_masks) == 0:
                return None

            # Generate labels.
            labels = [-100 if m == 0 else input_ids[i] for i, m in enumerate(assistant_masks)]
            # Force the first token's label to -100.
            if labels and labels[0] != -100:
                labels[0] = -100

            # Assemble the output.
            encoded = {}
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['loss_scale'] = None

            # Extract multimodal data: pass through every non-text key (same policy as the fallback path).
            _skip_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'assistant_masks'}
            for key, val in result.items():
                if key in _skip_keys:
                    continue
                if isinstance(val, (torch.Tensor, list)):
                    encoded[key] = val

            return encoded

        except Exception as e:
            logger.debug(f'_encode_via_chat_template failed: {e}')
            return None

    # ------------------------------------------------------------------
    # Shared _encode logic (used by all subclasses)
    # ------------------------------------------------------------------

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """Encode using the processor.

        Flow:
        1. _swift_prepare_inputs
        2. Try apply_chat_template + return_assistant_tokens_mask first
        3. Fall back to _build_text + TokenizeByRound when unsupported
        4. Compute loss_scale
        5. Assemble the output

        Note: non-thinking-prefix / thinking handling happens inside `_build_text` -> `_swift_encode`
        (idempotent), so it is not invoked here (matches legacy `_encode`).

        Reaching this method at all is opt-in (TemplateConfig.legacy_encode=False). There is no
        "delegate the body back to legacy" switch here any more: that shape LOOKED equivalent to
        using the legacy template but was not, because `replace_tag` and
        `Gemma3VisionTemplate._swift_prepare_inputs` sit on legacy's own call path and kept
        overriding it. build_template now derives from the legacy class instead (see
        `shifted_template_class`), so "use legacy's assembly" means "do not use this class".
        """
        inputs.messages = deepcopy(inputs.messages)
        self._swift_prepare_inputs(inputs)

        # Get the HF processor.
        hf_processor = self.processor
        tokenizer = self.tokenizer

        # --- Preferred path: return_assistant_tokens_mask ---
        if self.is_training and self.supports_assistant_tokens_mask:
            result = self._encode_via_chat_template(inputs, hf_processor, tokenizer)
            if result is not None:
                return result

        # --- Fallback path: _build_text + TokenizeByRound ---
        full_text = self._build_text(inputs)

        # Prepare processor kwargs. The meta-driven text already contains the needed bos (from meta
        # prefix or auto_add_bos), so never let the processor add another one.
        processor_kwargs = {'return_tensors': 'pt', 'add_special_tokens': False}
        media_kwargs = self._prepare_media_inputs(inputs, hf_processor)
        processor_kwargs.update(media_kwargs)
        if inputs.mm_processor_kwargs:
            processor_kwargs.update(inputs.mm_processor_kwargs)

        # Encode.
        full_encoded = hf_processor(text=[full_text], **processor_kwargs)
        input_ids = full_encoded['input_ids'][0].tolist()

        # Derive labels (TokenizeByRound).
        labels = None
        loss_scale = None
        if self.is_training:
            labels, assistant_ranges = self._build_labels_tokenize_by_round(input_ids, inputs, hf_processor,
                                                                            processor_kwargs)
            # Force the first token's label to -100 (matching the parent class).
            if labels and labels[0] != -100:
                labels[0] = -100
            # Compute loss_scale.
            loss_scale = self._compute_loss_scale(input_ids, labels, assistant_ranges, inputs)
            if loss_scale is not None and loss_scale[0] != 0:
                loss_scale[0] = 0

        # Assemble the output.
        encoded = {}
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale

        # Extract multimodal data: pass through every non-text key returned by the processor
        # (input_ids/attention_mask are handled separately). token_type_ids is not passed through
        # (legacy only generates it under specific conditions); subclasses handle it in _postprocess_encoded.
        _skip_keys = {'input_ids', 'attention_mask', 'token_type_ids'}
        for key, val in full_encoded.items():
            if key in _skip_keys:
                continue
            if isinstance(val, torch.Tensor):
                encoded[key] = val
            elif isinstance(val, list):
                encoded[key] = val

        # Subclass post-processing hook (adds token_type_ids/mm_token_type_ids etc.).
        self._postprocess_encoded(encoded, inputs, input_ids)

        # Clear labels in inference mode.
        if not self.is_training:
            for k in list(encoded.keys()):
                if k.endswith('labels') or k.endswith('loss_scale'):
                    encoded[k] = None

        return encoded


# ======================================================================
# Qwen VL Template (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL)
# ======================================================================


class QwenVLTemplate(Template):
    """Qwen VL series. Chat format comes from template_meta; only media handling is model-specific.

    Version differences (video placeholder/preprocessing):
    - v2 (qwen2_vl) / v2_5 (qwen2_5_vl): <video> -> <|vision_start|><|video_pad|><|vision_end|>
    - v3 (qwen3_vl / qwen3_5): <video> -> <|video_pad|>, requires video_metadata + do_sample_frames=False
    """

    @property
    def _qwen_version(self) -> str:
        tt = self.template_meta.template_type
        if tt == 'qwen2_vl':
            return 'v2'
        if tt == 'qwen2_5_vl':
            return 'v2_5'
        return 'v3'  # qwen3_vl / qwen3_5

    def replace_tag(self, media_type, index, inputs):
        if media_type == 'image':
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        if media_type == 'video':
            if self._qwen_version == 'v3':
                return ['<|video_pad|>']
            return ['<|vision_start|><|video_pad|><|vision_end|>']
        return super().replace_tag(media_type, index, inputs)

    def _prepare_media_inputs(self, inputs: StdTemplateInputs, hf_processor) -> dict:
        """Qwen: pre-resize via fetch_image/fetch_video, so pass do_resize=False."""
        kwargs = {}
        images = inputs.images if inputs.images else None
        videos = inputs.videos if inputs.videos else None
        version = self._qwen_version
        if images is not None:
            kwargs['images'] = images
            kwargs['do_resize'] = False
        if videos is not None:
            from qwen_vl_utils import fetch_video, vision_process
            fetch_kwargs = {}
            if version == 'v3':
                fetch_kwargs['image_patch_size'] = hf_processor.image_processor.patch_size
                fetch_kwargs['return_video_metadata'] = True
            processed = []
            fps_list = []
            metadata = []
            for v in videos:
                video_inputs = {'video': v}
                if isinstance(v, list):  # list of frames
                    video_inputs['sample_fps'] = vision_process.FPS
                fv, vk = fetch_video(video_inputs, return_video_sample_fps=True, **fetch_kwargs)
                if version == 'v3':
                    fv, vmeta = fv  # v3: fetch_video returns (video, metadata)
                    metadata.append(vmeta)
                if isinstance(fv, torch.Tensor):
                    fv = fv.to(torch.uint8)
                processed.append(fv)
                fps_list.append(vk)
            kwargs['videos'] = processed
            kwargs['do_resize'] = False
            if version == 'v3':
                kwargs['video_metadata'] = metadata
                kwargs['do_sample_frames'] = False
            # Stash fps so second_per_grid_ts can be computed the legacy way in _postprocess_encoded
            # (do not rely on the processor's fps->second_per_grid_ts, whose convention differs from legacy).
            self._video_fps = fps_list
        return kwargs

    def _postprocess_encoded(self, inputs_encoded, inputs: StdTemplateInputs, input_ids: List[int]) -> None:
        # Qwen2.5-VL(v2_5): second_per_grid_ts = temporal_patch_size / fps (list, matching legacy).
        if inputs.videos and self.template_meta.template_type == 'qwen2_5_vl':
            tps = self.processor.image_processor.temporal_patch_size
            fps_list = getattr(self, '_video_fps', None)
            if fps_list:
                inputs_encoded['second_per_grid_ts'] = [tps / f for f in fps_list]


# ======================================================================
# Media-tag-only subclasses: chat format from template_meta, placeholder from _media_tag_map
# ======================================================================


class Gemma4Template(Template):
    """Gemma4 (<bos><|turn>role...<turn|>). Format from meta; media via _media_tag_map."""

    _media_tag_map = {'<image>': '<|image|>', '<video>': '<|video|>', '<audio>': '<|audio|>'}


class InternVLTemplate(Template):
    """InternVL3.5+ (ChatML). <image> -> <IMG_CONTEXT>\\n."""

    _media_tag_map = {'<image>': '<IMG_CONTEXT>\n'}


class LLaVATemplate(Template):
    """LLaVA-OneVision (ChatML). <image> -> <image>\\n (matches legacy replace_tag)."""

    _media_tag_map = {'<image>': '<image>\n'}


# ======================================================================
# Gemma3-Vision Template
# ======================================================================


class Gemma3VisionTemplate(Template):
    """Gemma3-Vision. Chat format from template_meta; system merged into the first user turn;
    adds token_type_ids marking image tokens.
    """

    _media_tag_map = {'<image>': '<start_of_image>'}

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._swift_prepare_inputs(inputs)
        # Match Gemma3Template._swift_encode: merge system into the first user, strip assistant newlines.
        if inputs.system is not None and inputs.messages and isinstance(inputs.messages[0]['content'], str):
            system = inputs.system
            inputs.system = None
            inputs.messages[0]['content'] = system + '\n\n' + inputs.messages[0]['content']
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                message['content'] = message['content'].strip('\n')

    def _prepare_media_inputs(self, inputs: StdTemplateInputs, hf_processor) -> dict:
        kwargs = {}
        if inputs.images:
            kwargs['images'] = inputs.images
        return kwargs

    def _postprocess_encoded(self, encoded: Dict[str, Any], inputs: StdTemplateInputs, input_ids: List[int]) -> None:
        if not inputs.images:
            return
        import numpy as np
        image_token_id = getattr(self.processor, 'image_token_id', None)
        if image_token_id is None:
            return
        arr = np.array(input_ids)
        tti = np.zeros_like(arr)
        tti[arr == image_token_id] = 1
        encoded['token_type_ids'] = tti.tolist()


# ======================================================================
# Idefics3 Template
# ======================================================================
# The chat format comes from the (fixed) idefics3 TemplateMeta, which must match the official
# apply_chat_template: 'User: '/'Assistant: ' (trailing space after the colon) and a last turn ending
# with '<end_of_utterance>\n'. See swift/template/templates/idefics3.py.


class Idefics3Template(Template):
    """Idefics3. Format from template_meta; the processor expands <image> itself."""

    # _media_tag_map empty -> keep <image>; the processor expands it.


# ======================================================================
# Pixtral Template
# ======================================================================


class PixtralTemplate(Template):
    """Pixtral (<s>{system}[INST]{q}[/INST]{r}</s>). Format from meta; <image> -> [IMG]."""

    _media_tag_map = {'<image>': '[IMG]'}


# ======================================================================
# Deferred families (NOT in the mapping)
# ======================================================================
# - llava1_5 / llava1_6_mistral etc. (SentencePiece): legacy tokenizes each context separately and
#   inserts a dummy leading space per context (e.g. ' I' -> '▁I'), which a single whole-string
#   processor tokenize cannot reproduce -> input_ids differ. Keep legacy.
# - qwen2_audio: official examples (transformers 4.45~4.49, processor(text, audios=...)) keep a single
#   <|AUDIO|> and expand inside the model forward; swift legacy already matches. Keep legacy.

# ======================================================================
# Mapping table
# ======================================================================

PROCESSOR_TEMPLATE_MAPPING = {
    # Qwen VL series
    'qwen2_vl': QwenVLTemplate,
    'qwen2_5_vl': QwenVLTemplate,
    'qwen3_vl': QwenVLTemplate,
    'qwen3_5': QwenVLTemplate,  # Qwen3.5-VL uses template 'qwen3_5'
    # Gemma4
    'gemma4': Gemma4Template,
    'gemma4_nothinking': Gemma4Template,
    # Gemma3-Vision
    'gemma3_vision': Gemma3VisionTemplate,
    # Pixtral
    'pixtral': PixtralTemplate,
    # Idefics3 (aligned with official apply_chat_template)
    'idefics3': Idefics3Template,
    # InternVL
    'internvl_hf': InternVLTemplate,
    # LLaVA
    'llava_onevision_hf': LLaVATemplate,
    'llava_onevision1_5': LLaVATemplate,
}
