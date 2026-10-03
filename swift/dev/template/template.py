# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List, Optional

from swift.dev.utils import get_logger
from swift.template.base import Template as LegacyTemplate

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

    def encode(self,
               inputs,
               return_template_inputs: bool = False,
               return_length: bool = False,
               add_generation_prompt: Optional[bool] = None):
        """Encode, then shift labels to next-token alignment (training mode only).

        ``twinkle.vLLMSampler`` passes ``add_generation_prompt`` explicitly. The legacy swift
        template derives the same decision from the final message role, so the compatibility keyword
        is accepted here while the underlying encoder remains the single source of prompt formatting.

        vLLM-mode guard: the next-token shift must fire ONLY in training modes. In
        inference/rollout modes (vllm/lmdeploy/sglang/transformers) legacy `is_training` is False and
        `_encode` clears labels to None, so today the `labels is not None` check already skips the
        shift. We ALSO gate on `self.is_training` explicitly so a rollout input can never be silently
        shifted even if a future mode were to emit labels.

        Task guard: only `causal_lm` labels are per-token. See ``_NO_SHIFT_TASK_TYPES``.
        """
        del add_generation_prompt
        encoded = super().encode(inputs, return_template_inputs=return_template_inputs, return_length=return_length)
        if (self.is_training and getattr(self, 'task_type', 'causal_lm') not in self._NO_SHIFT_TASK_TYPES
                and isinstance(encoded, dict) and not encoded.get(self.SHIFTED_KEY)):
            # Every label key the encoders emit needs the same next-token shift, not just the SFT bare
            # `labels`. The preference path (`_rlhf_encode`) emits `chosen_labels`/`rejected_labels`
            # instead of a bare `labels` (run_dpo._encode_pair strips the prefix back to `labels` later),
            # so shifting only the bare key left dpo/kto/cpo/orpo/simpo feeding ALIGNED labels to
            # twinkle's no-shift selective_log_softmax -- log P(x_i | x_<=i), scoring the model on a token
            # it can already see. Each `<side>_labels` carries a matching `<side>_loss_scale` shifted in
            # step; seq_cls/rm is excluded by the task guard above (its per-side labels are popped).
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
        """Return the token ids consumed by vLLM for text-only dev rollout."""
        return input_ids

    def concat_input_feature(self,
                             prompt_input_feature,
                             new_tokens: List[int],
                             *,
                             appended_as: str = 'completion',
                             tool_calls: Optional[List[Dict[str, Any]]] = None):
        """Append one sampled turn to an already-encoded prefix, keeping the token account whole.

        Mirrors twinkle's native ``concat_input_feature``: unroll the prefix's ``labels`` /
        ``completion_mask`` from output order back to input order, append the new tokens (trainable
        for a completion, masked for ``context``), then re-roll through ``_invoke_post_pipeline`` --
        the exact inverse of what the next ``observe`` / ``append_ids`` does, so the two stay aligned
        turn after turn.

        The unroll-append-reroll is load-bearing in the MULTI-turn path, which the single-turn
        shortcut this replaced silently broke. There the prefix already carries earlier turns'
        trainable labels plus a ``completion_mask`` the ledger's ``audit`` zips against ``labels``
        position-by-position. Rebuilding ``labels`` as ``[-100] * prompt + response`` wiped every
        earlier turn's labels, and leaving ``completion_mask`` untouched let it fall behind the
        growing tokens -- so the second assistant turn of any tool episode died in ``audit`` with a
        ``completion_mask/labels misaligned`` length mismatch (the gap that surfaced only once
        ``_invoke_post_pipeline`` let the observe path run at all). Single-turn is unaffected either
        way: ``samples_from_responses`` rebuilds ``encoded`` from ``prompt_token_ids`` and never reads
        this feature's labels, and an opening prompt is all-masked so both forms agree.

        The message append is not cosmetic: the multi-turn ledger adopts this feature wholesale
        (``record`` does ``_pif = new_input_feature``), and every consumer reads the episode back off
        ``messages`` -- ``run_infer`` / the TUI take the reply from the last assistant turn, and the
        rollout loop's ``last_msg`` is ``messages[-1]``. A reply that parses as a call is stored with
        the markup cleaned out of ``content`` and the structured calls in their own field; a sampler
        that already parsed ``tool_calls`` passes them so the text is not re-parsed.
        ``appended_as='context'`` masks the turn out of the labels and the completion_mask (history a
        later turn sees but no loss may touch). Text-only: multimodal ``mm_token_type_ids`` padding is
        left to ``append_ids``, the only grower a dev multimodal episode uses.
        """
        import copy

        result = copy.deepcopy(prompt_input_feature)
        prompt_ids = list(result['input_ids'])
        response_ids = list(new_tokens)
        scored = appended_as != 'context'

        # Unroll the prefix's labels (output -> input order). An opening prompt encoded for generation
        # has none, so it is all-masked; a multi-turn prefix carries earlier turns' trainable labels,
        # which the circular unroll recovers exactly because position 0 is always a masked prompt token
        # (so the wrapped value is -100 either way, matching dev's non-circular encode shift).
        labels = list(result.get('labels') or [])
        labels = (labels[-1:] + labels[:-1]) if labels else [-100] * len(prompt_ids)
        # completion_mask lives on labels' index space, so it unrolls in step. Derived from labels when
        # the prefix predates the field (the single-turn opening), which keeps old and new trajectories
        # equivalent because the trainable positions were exactly ``labels != -100``.
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
        # append_ids runs, so labels and completion_mask stay the same length and in the same order.
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

        twinkle's multi-turn ledger grows an episode with ``append_ids`` / ``extend_with_bridge``
        (a tool observation, a bridge to the next generation prompt). After splicing the new ids in
        INPUT order, ``append_ids`` calls ``template._invoke_post_pipeline([result])`` to (1) enforce
        ``max_length``, (2) refresh the sequence-aligned fields and (3) roll ``labels`` (and
        ``completion_mask``) from input order back into the next-token/output order that the rest of
        twinkle -- and dev's own ``concat_input_feature`` and ``trajectory_to_rollout_sample`` -- read.
        A swift legacy template has none of twinkle's pipeline stages, so the first tool observation
        died with ``AttributeError: 'Shifted*Template' object has no attribute '_invoke_post_pipeline'``
        (the third gap in the same "DevMixin did not fully bridge the twinkle Template contract" seam).

        The roll MUST be circular (``x[1:] + x[:1]``, i.e. ``np.roll(-1)``) so it is the exact inverse
        of the unroll ``append_ids``/``_prefix_completion_mask`` do on the next turn (``x[-1:] + x[:-1]``)
        and of dev's ``_input_order_labels``. That is also what makes it agree with dev's
        ``concat_input_feature`` shift, which is non-circular but identical here because position 0 is
        always a masked prompt token (label -100 / mask 0). ``labels`` and ``completion_mask`` are rolled
        TOGETHER because the ledger's ``audit`` zips them position-by-position to count policy tokens.

        Only the ordering and the max_length verdict are load-bearing downstream: the vLLM sampler
        re-feeds ``input_ids``, and ``trajectory_to_rollout_sample`` reads ``labels``/``completion_mask``.
        ``attention_mask``/``length`` are refreshed for faithfulness; ``position_ids`` is left to the
        engine (vLLM derives its own, and dev has no ``set_mm_position_ids``). ``max_length`` overflow
        under the 'delete' strategy drops the feature (empty list -> ``append_ids`` returns None ->
        ``ledger.observe`` False -> the engine flags ``truncated``); other strategies keep it whole,
        since dev bounds rollout length with ``max_trajectory_tokens`` rather than a hard re-raise here.
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

    # twinkle's AGENT contract (the rollout half; the Model half is encode/batch_encode above): a multi-turn
    # tool loop parses tool calls off the model's decoded text THROUGH the template -- twinkle_agentic's
    # rollout ledger calls template.parse_tool_call / tool_call_errors, and its endpoint path calls
    # clean_tool_call. twinkle's native Template answers these from its ToolCallRegistry; a swift legacy
    # template has none of them, so the first tool turn dies with AttributeError. Delegate to the SAME
    # registry twinkle uses: its parsers are format detectors, and HermesQwenParser matches the
    # <tool_response> markup swift's own Qwen agent templates render, so a dev template parses its own
    # tool calls correctly. Imported lazily to keep this module cheap to import.
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
