# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List, Optional

from swift.dev.utils import get_logger
from swift.template.base import Template as LegacyTemplate

logger = get_logger()


class DevMixin:
    """dev's entire delta over a legacy Template: the twinkle contract, nothing else.

    Encoding stays in swift's legacy Template; this mixin adds the twinkle Model contract (next-token
    labels, ``batch_encode``, agent tool parsing, multi-turn token bookkeeping). Mixed into the real
    legacy class by :func:`shifted_template_class`, preceding it in the MRO.
    """

    # Records that a sample is already shifted, so a second encode cannot shift twice.
    SHIFTED_KEY = '_labels_shifted'

    @staticmethod
    def _shift_labels_next_token(labels: List[int]) -> List[int]:
        """Explicit next-token shift: out[i]=labels[i+1], out[-1]=-100 (NOT circular roll)."""
        if not labels:
            return labels
        return list(labels[1:]) + [-100]

    # `labels` are NOT per-token targets on these tasks (embedding: 1.0 marks the group anchor InfonceLoss
    # locates via nonzero(labels); reranker/seq_cls: per-sequence scores), so the shift must not fire.
    _NO_SHIFT_TASK_TYPES = frozenset({'embedding', 'reranker', 'generative_reranker', 'seq_cls'})

    def encode(self,
               inputs,
               return_template_inputs: bool = False,
               return_length: bool = False,
               add_generation_prompt: Optional[bool] = None):
        """Encode, then shift labels to next-token alignment (training modes only).

        twinkle's forward computes logps via no-shift selective_log_softmax, so twinkle expects
        NEXT-token labels; legacy emits aligned labels, so the shift happens here ("whoever encodes,
        shifts"). Guards: training mode only, task guard (``_NO_SHIFT_TASK_TYPES``), ``SHIFTED_KEY``.
        """
        # v5 schema gate: validate a raw standard row BEFORE the legacy encoder (missing keys
        # otherwise surface as cryptic crashes). Local import keeps the loader stack out of load time.
        if isinstance(inputs, dict) and 'input_ids' not in inputs:
            from swift.dev.dataset.schema import validate_standard_row
            validate_standard_row(inputs, getattr(self, 'task_type', None), getattr(self, 'mode', None))
        del add_generation_prompt
        encoded = super().encode(inputs, return_template_inputs=return_template_inputs, return_length=return_length)
        if (self.is_training and getattr(self, 'task_type', 'causal_lm') not in self._NO_SHIFT_TASK_TYPES
                and isinstance(encoded, dict) and not encoded.get(self.SHIFTED_KEY)):
            # Shift EVERY label key, not just bare `labels` (the preference path emits
            # chosen_/rejected_ + matching <side>_loss_scale) -- else dpo/kto/cpo/orpo feed aligned labels.
            shifted = False
            for label_key in ('labels', 'chosen_labels', 'rejected_labels'):
                if encoded.get(label_key) is None:
                    continue
                encoded[label_key] = self._shift_labels_next_token(list(encoded[label_key]))
                scale_key = label_key.replace('labels', 'loss_scale')
                if encoded.get(scale_key) is not None:
                    encoded[scale_key] = list(encoded[scale_key][1:]) + [0.0]
                shifted = True
            if shifted:
                encoded[self.SHIFTED_KEY] = True
        return encoded

    def get_vllm_input_ids(self, input_ids):
        """Return the token ids vLLM consumes: collapse each media-pad run back to ONE placeholder.

        A VLM template's ``_encode`` EXPANDS each placeholder to N media-pad tokens (the training form,
        frozen against the pixel grid it ships); vLLM expects a single placeholder and re-expands it
        from its own grid -- handing it the expanded run double-expands and breaks the sampler's
        length guard. No ``image_token_id`` means text-only, ids untouched.
        """
        image_token_id = getattr(self, 'image_token_id', None)
        if image_token_id is None:
            return input_ids
        media_pad_ids = {image_token_id}
        video_token_id = getattr(self, 'video_token_id', None)
        if video_token_id is not None:
            media_pad_ids.add(video_token_id)
        result: List[int] = []
        prev_pad = None
        for token in input_ids:
            if token in media_pad_ids:
                if token == prev_pad:
                    continue  # collapse a consecutive same-media pad run to its first token
                prev_pad = token
            else:
                prev_pad = None
            result.append(token)
        return result

    def concat_input_feature(self,
                             prompt_input_feature,
                             new_tokens: List[int],
                             *,
                             appended_as: str = 'completion',
                             tool_calls: Optional[List[Dict[str, Any]]] = None):
        """Append one sampled turn to an encoded prefix, keeping the token account whole.

        Mirrors twinkle's native concat_input_feature: unroll labels/completion_mask to input order,
        append the new tokens (trainable for a completion, masked for ``context``), re-roll through
        ``_invoke_post_pipeline``. The unroll-append-reroll is load-bearing multi-turn: a single-turn
        shortcut dies in the ledger's ``audit`` on the second tool turn.
        """
        import copy

        result = copy.deepcopy(prompt_input_feature)
        prompt_ids = list(result['input_ids'])
        response_ids = list(new_tokens)
        scored = appended_as != 'context'

        # Unroll the prefix's labels (output -> input order); an opening prompt's are all-masked,
        # and the circular unroll is exact because position 0 is always a masked prompt token.
        labels = list(result.get('labels') or [])
        labels = (labels[-1:] + labels[:-1]) if labels else [-100] * len(prompt_ids)
        # completion_mask unrolls in step on labels' index space; derived from labels when the prefix
        # predates the field (the trainable positions were exactly ``labels != -100``).
        mask = result.get('completion_mask')
        if mask is None:
            mask = [0 if label == -100 else 1 for label in labels]
        else:
            mask = list(mask)
            mask = mask[-1:] + mask[:-1]
            if len(mask) != len(prompt_ids):
                raise ValueError(f'prefix completion_mask has {len(mask)} entries for {len(prompt_ids)} '
                                 'input_ids; appending would misalign every position after it.')

        result['input_ids'] = prompt_ids + response_ids
        result['labels'] = labels + (response_ids if scored else [-100] * len(response_ids))
        result['completion_mask'] = mask + ([1] * len(response_ids) if scored else [0] * len(response_ids))
        result[self.SHIFTED_KEY] = True
        # Re-roll into output order and refresh the sequence-aligned fields, the same post pipeline
        # append_ids runs, so labels and completion_mask stay aligned in length and order.
        result = self._invoke_post_pipeline([result])[0]

        messages = result.get('messages')
        if messages is not None:
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if tool_calls is None:
                parsed = self.parse_tool_call(response_text) or []
                content_text = self.clean_tool_call(response_text) if parsed else response_text
            else:
                parsed = list(tool_calls)
                content_text = response_text
            assistant_message: Dict[str, Any] = {'role': 'assistant', 'content': content_text}
            if parsed:
                assistant_message['tool_calls'] = parsed
            messages.append(assistant_message)
            result['messages'] = messages
        return result

    def _invoke_post_pipeline(self, input_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """twinkle's post-encode pipeline, reimplemented on dev's label convention.

        twinkle's ledger grows an episode with append_ids/extend_with_bridge, then calls this to (1)
        enforce max_length, (2) refresh sequence-aligned fields, (3) roll labels/completion_mask back
        to output order. The roll MUST be circular (exact inverse of the next turn's unroll) and roll
        labels+completion_mask TOGETHER (audit zips them). Only 'delete' drops an overlong feature.
        """
        max_length = getattr(self, 'max_length', None)
        strategy = getattr(self, 'truncation_strategy', 'raise')
        out: List[Dict[str, Any]] = []
        for feature in input_features:
            input_ids = feature.get('input_ids')
            if input_ids is None:
                out.append(feature)
                continue
            if max_length and len(input_ids) > max_length and strategy == 'delete':
                continue  # dropped: signals graceful truncation to append_ids/observe
            feature['attention_mask'] = [1] * len(input_ids)
            feature['length'] = len(input_ids)
            if feature.get('labels') is not None:
                labels = list(feature['labels'])
                feature['labels'] = labels[1:] + labels[:1]
            if feature.get('completion_mask') is not None:
                mask = list(feature['completion_mask'])
                feature['completion_mask'] = mask[1:] + mask[:1]
            out.append(feature)
        return out

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_encode(self, trajectories, add_generation_prompt: bool = False, **kwargs):
        """twinkle's batch entry point, delegated row-by-row to swift's ``encode``.

        Deliberately thin: routing the model through a swift template means ONE encode implementation
        produces the training tokens. ``add_generation_prompt`` is a twinkle inference concept with no
        swift counterpart; it is accepted for signature compatibility and rejected when set, as is the
        columnar-dict form the Model never passes.
        """
        if add_generation_prompt:
            raise NotImplementedError(
                'batch_encode(add_generation_prompt=True) is not supported on a swift template: the '
                'generation prompt is decided by the template mode (set_mode), not per call.')
        if isinstance(trajectories, dict):
            raise NotImplementedError('batch_encode expects a list of trajectories, not a columnar dict.')
        return [self.encode(dict(trajectory), **kwargs) for trajectory in trajectories]

    # twinkle's AGENT contract (rollout half): parse_tool_call / tool_call_errors / clean_tool_call
    # delegate to the SAME registry twinkle uses (HermesQwenParser matches swift's Qwen markup).
    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        from twinkle.template.tools import ToolCallRegistry
        parser = ToolCallRegistry.detect_first(decoded or '')
        return parser.parse(decoded) if parser else []

    def clean_tool_call(self, decoded: str) -> str:
        from twinkle.template.tools import ToolCallRegistry
        parser = ToolCallRegistry.detect_first(decoded or '')
        return parser.clean(decoded) if parser else (decoded or '').rstrip()

    def tool_call_errors(self, decoded: str) -> List[str]:
        from twinkle.template.tools import ToolCallRegistry
        parser = ToolCallRegistry.detect_first(decoded or '')
        return parser.parse_errors(decoded) if parser else []


# Cache keyed by legacy class: one derived class per family, so `isinstance` stays meaningful across
# templates and the class is created once.
_SHIFTED_CLASSES: Dict[type, type] = {}

# Reverse index (derived class NAME -> legacy base), used by this module's __getattr__ to rebuild a
# class when pickle looks it up by name in a process that never created it.
_SHIFTED_BASES: Dict[str, type] = {}


def shifted_template_class(base: type) -> type:
    """``base`` + the next-token shift, as a class -- deriving, not ``__class__`` replacement.

    ``Template.from_template`` overwrites ``__class__`` and drops every method the family overrode;
    deriving keeps them all. The class is registered in this module's globals AND rebuilt on demand
    by ``__getattr__`` so it stays PICKLABLE under datasets.map's num_proc > 1 workers (pickle
    resolves classes by module + qualname).
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
    """Rebuild a ``Shifted<Family>`` class on first lookup, so unpickling works in a fresh process.

    pickle stores a dynamically created class as module + qualname and resolves it with getattr on
    the imported module; a datasets.map worker is a new interpreter where the name is absent, and
    without this hook the load fails with AttributeError -- only under num_proc > 1.
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
    """Locate a legacy Template subclass by its __name__, for __getattr__'s rebuild path."""
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
