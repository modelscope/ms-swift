# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loader that attaches a typed-decision scoring head onto a base LM (`decision` task_type).

The `decision` task_type does not touch `register.py::get_model`'s dispatch: it falls through to
the plain `patch_automodel` branch, so `super().get_model()` loads the stock base LM (Qwen3-VL /
Qwen3.5 ...). `ScoringModelLoader` then attaches a `ScoringHead` and patches `forward` so a single
pass returns a `ScoringOutput` of per-option logits (see `swift/model/decision_head.py`).

Per plan decision B, LoRA is NOT handled here -- it goes through swift's standard `--adapters` /
PEFT path. Per plan decision F, the head's factory weights are a separate artifact loaded strictly
by the concrete head's `from_pretrained`; the base LM checkpoint is left untouched.
"""
from contextlib import contextmanager
from functools import wraps
from typing import Optional

from transformers import PreTrainedModel, PretrainedConfig

from swift.model.decision_head import ClefJointSchemaHead, JevVerbalizerHead, OmniJevHead, ScoringHead
from swift.model.omnijev_branch import (OPT_CLOSE, OPT_OPEN, add_option_tokens, branch_questions, find_rope_owner,
                                        is_hybrid, prefix_inputs, rope_positions, rows_from_ids)
from swift.model.patcher import patch_module_forward
from swift.template import TemplateType
from swift.utils import Processor, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model
from .qwen import Qwen3_5Loader

logger = get_logger()


def _scoring_head_forward(self, *args, head: ScoringHead, origin_forward, **kwargs):
    """Patched forward: run the base LM, then score every question over its own option set.

    `decision_meta` is injected by the template's `_post_encode` (multimodal hook) or passed
    straight through (text path); it is popped here so the base forward never sees it. `labels`
    are popped and ignored -- the loss is computed by `ScoringTrainer` via `compute_loss_func`
    (a `ScoringLoss`), not inside forward, so this returns a `ScoringOutput` with `loss=None`.

    `output_hidden_states=True` gives `hidden_states[-1]` (the last-norm hidden state) for the
    span-pooling heads (Clef / OmniJev). `lm_logits` is only forwarded when the head asks for it
    (`needs_lm_logits`, JEV's verbalizer columns). Known perf tradeoff: because we call the base
    LM's own forward, its lm_head projection still runs even for span-pooling heads that ignore
    `lm_logits`; skipping it (calling the backbone directly) is an optimization to verify against a
    real model in the run phase, not done here to keep the base path architecture-agnostic.
    """
    meta = kwargs.pop('decision_meta', None)
    kwargs.pop('labels', None)
    kwargs['output_hidden_states'] = True
    kwargs['return_dict'] = True
    output = origin_forward(*args, **kwargs)
    hidden_states = output.hidden_states[-1]
    lm_logits = output.logits if head.needs_lm_logits else None
    return head(hidden_states, lm_logits, meta)


class ScoringModelLoader(ModelLoader):
    """Base loader for the three decision models.

    Subclasses set `head_cls` to a `ScoringHead` subclass (whose `from_pretrained` strict-loads the
    factory head weights), or override `build_head` for a non-standard construction.
    """
    head_cls: Optional[type] = None

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, config, processor, model_kwargs)
        head = self.build_head(model, model_dir, config)
        self.attach_scoring_head(model, head)
        return model

    def build_head(self, model: PreTrainedModel, model_dir: str, config: PretrainedConfig) -> ScoringHead:
        if self.head_cls is None:
            raise NotImplementedError('ScoringModelLoader subclasses must set `head_cls` or override `build_head`.')
        return self.head_cls.from_pretrained(model_dir, model=model, config=config)

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        """Register the head as a submodule, match the base LM's dtype/device, wire the lexical-prior
        reference if needed, mark its params trainable, and patch `forward`."""
        head = self._match_head_dtype_device(model, head)
        model.scoring_head = head
        if head.needs_output_embeddings:
            head.set_output_embeddings(model.get_output_embeddings())
        # Plan decision C: base frozen + LoRA + head jointly trained. Mark head params trainable so
        # they survive PEFT's freeze of the base; review-mode must confirm PEFT does not re-freeze them.
        for p in head.parameters():
            p.requires_grad_(True)

        origin_forward = model.forward

        @wraps(origin_forward.__func__ if hasattr(origin_forward, '__func__') else origin_forward)
        def new_forward(self, *args, **kwargs):
            return _scoring_head_forward(self, *args, head=head, origin_forward=origin_forward, **kwargs)

        patch_module_forward(model, new_forward)

    @staticmethod
    def _match_head_dtype_device(model: PreTrainedModel, head: ScoringHead) -> ScoringHead:
        """Move the head to the base LM's device and (unless `head.keep_fp32`) cast its floating
        params/buffers to the base dtype. Integer buffers (e.g. JEV's verbalizer_ids) are left
        untouched by `Module.to(dtype=...)`; `keep_fp32` heads (JEV's fp32 calibration bias) are only
        moved, never downcast."""
        ref_weight = None
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and getattr(output_embeddings, 'weight', None) is not None:
            ref_weight = output_embeddings.weight
        else:
            ref_weight = next((p for p in model.parameters()), None)
        if ref_weight is not None:
            if head.keep_fp32:
                head = head.to(device=ref_weight.device)
            else:
                head = head.to(device=ref_weight.device, dtype=ref_weight.dtype)
        return head


class JevModelLoader(ScoringModelLoader, Qwen3_5Loader):
    """JEV loader: load the Qwen3.5-VL base exactly as `Qwen3_5Loader` does, then attach the fp32
    `JevVerbalizerHead` and patch `forward`.

    MRO is JevModelLoader -> ScoringModelLoader -> Qwen3_5Loader -> Qwen3VLLoader -> ... -> ModelLoader,
    so `ScoringModelLoader.get_model`'s `super().get_model(...)` resolves to `Qwen3_5Loader.get_model`
    -- the base loader that pins `auto_model_cls=Qwen3_5ForConditionalGeneration` and applies the
    keep-in-fp32 / vision-hook patches. The base is therefore loaded identically to a plain Qwen3.5
    model, and the head is attached afterwards. This is the plan §2.2 Loader-trap resolution: reuse
    the base loader via multiple inheritance instead of re-hardcoding `auto_model_cls`.
    """
    head_cls = JevVerbalizerHead


register_model(
    ModelMeta(
        MLLMModelType.jev,
        [ModelGroup([
            Model('autotrust/JEV-27B-VL', 'autotrust/JEV-27B-VL'),
        ], TemplateType.jev)],
        JevModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        # `architectures` is intentionally EMPTY. JEV's factory config.json declares
        # `Qwen3_5ForConditionalGeneration`, which `MLLMModelType.qwen3_5` already claims for
        # architecture-based auto-detection (model_meta.py::_get_arch_mapping -> get_matched_model_types).
        # Re-declaring it here would make an *unnamed* Qwen3.5 checkpoint resolve to TWO model_types and
        # raise "Multiple possible types found" -- a regression to the existing qwen3_5 path, which the
        # plan forbids (§2.2 note 91: zero spillover). JEV is matched by NAME (autotrust/JEV-27B-VL) or
        # an explicit `--model_type jev`; both bypass arch detection, so an empty list is safe and keeps
        # the qwen3_5 arch bucket single-valued.
        architectures=[],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'decision']))


class ClefModelLoader(ScoringModelLoader, Qwen3_5Loader):
    """Clef loader: load the merged Qwen3.5-VL backbone exactly as `Qwen3_5Loader` does, then attach
    the parameterised `ClefJointSchemaHead` (strict-loaded from `joint_head.safetensors`) and patch
    `forward`.

    MRO is ClefModelLoader -> ScoringModelLoader -> Qwen3_5Loader -> ... -> ModelLoader, so
    `ScoringModelLoader.get_model`'s `super().get_model(...)` resolves to `Qwen3_5Loader.get_model`
    (same reuse-via-multiple-inheritance as `JevModelLoader`). Clef ships a MERGED backbone (no LoRA),
    so only the head is a separate artifact -- plan decision B/F. `needs_output_embeddings=True` makes
    `attach_scoring_head` wire the base output embeddings into the head for its lexical prior.
    """
    head_cls = ClefJointSchemaHead


register_model(
    ModelMeta(
        MLLMModelType.clef,
        [ModelGroup([
            Model('Cloudflare/clef', 'Cloudflare/clef'),
        ], TemplateType.clef)],
        ClefModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        # `architectures` is intentionally EMPTY -- same reason as JEV: Clef's factory config.json also
        # declares `Qwen3_5ForConditionalGeneration`, which MLLMModelType.qwen3_5 already claims for
        # arch-based auto-detection. Re-declaring it would make an unnamed Qwen3.5 checkpoint resolve to
        # multiple model_types and raise "Multiple possible types found" (plan §2.2 note 91: zero
        # spillover). Clef is matched by NAME (Cloudflare/clef) or an explicit `--model_type clef`.
        architectures=[],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'decision']))


@contextmanager
def _grad_ckpt_disabled(backbone_model):
    """HF forces `use_cache=False` inside the backbone forward whenever `gradient_checkpointing and
    training`, which returns `past_key_values=None`. The OmniJev branch is cache-based: its prefix pass
    MUST hand a KV/recurrent cache to every option row (`expand_cache`), so gradient checkpointing at the
    model level is fundamentally incompatible with it (the branch saves memory with its own chunk
    recomputation, MSO_BRANCH_CKPT, instead). Temporarily clear the flag on every submodule that carries
    it and restore the exact prior values on exit, so nothing outside the branch observes the change and
    a training run that set `--gradient_checkpointing true` still gets a correct cache."""
    touched = [m for m in backbone_model.modules() if getattr(m, 'gradient_checkpointing', False)]
    for m in touched:
        m.gradient_checkpointing = False
    try:
        yield
    finally:
        for m in touched:
            m.gradient_checkpointing = True


def _omnijev_branch_forward(self, *args, head, origin_forward, backbone_model, tokenizer, open_ids, close_ids,
                            **kwargs):
    """Patched forward for OmniJev (decision G1): the two-stage cached BRANCH, not a single pass. The
    hybrid Qwen3.5-4B backbone carries recurrent state along the sequence, so a block-diagonal mask
    cannot isolate options; instead the shared prefix (chat head + image + instruction) is forwarded ONCE
    per record and every option block continues a COPY of that cache as its own row, which restores exact
    order-invariant isolation (swift/model/omnijev_branch.py, a port of mso/branch.py). decision_meta
    (from OmniJevTemplate) carries, per question, the option-marker positions in the real (unpadded) token
    space, and per record the media counts used to slice the collator-concatenated pixel tensors. labels
    are popped (ScoringTrainer computes the loss); returns a ScoringOutput via head.forward_features."""
    meta = kwargs.pop('decision_meta', None)
    kwargs.pop('labels', None)
    if meta is None:
        raise ValueError('OmniJev branch forward requires decision_meta from OmniJevTemplate._encode.')
    input_ids = kwargs.get('input_ids')
    if input_ids is None:
        raise ValueError(
            'OmniJev branch forward requires input_ids; OmniJevTemplate._post_encode must NOT convert the '
            'batch to inputs_embeds (the branch re-runs the backbone on the real token ids + pixels).')
    attention_mask = kwargs.get('attention_mask')
    pixel_values = kwargs.get('pixel_values')
    image_grid_thw = kwargs.get('image_grid_thw')
    mm_token_type_ids = kwargs.get('mm_token_type_ids')
    dev = input_ids.device
    B, T = input_ids.shape[0], input_ids.shape[1]
    left_padding = bool(meta.get('left_padding', False))
    seq_lens = meta.get('seq_lens')
    counts = [int(c) for c in meta['counts']]
    option_markers = meta['option_markers']
    mm_counts = meta.get('mm_counts') or [{'image': 0, 'video': 0}] * B
    lm_head = backbone_model.get_output_embeddings()
    rope_owner = find_rope_owner(backbone_model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    img_off = 0
    patch_off = 0
    q_cursor = 0
    per_question_features = []
    for r in range(B):
        if attention_mask is not None:
            sl = int(attention_mask[r].sum())
        elif seq_lens is not None:
            sl = int(seq_lens[r])
        else:
            sl = T
        if left_padding:
            real_ids = input_ids[r, T - sl:]
            real_mm = mm_token_type_ids[r, T - sl:] if mm_token_type_ids is not None else None
        else:
            real_ids = input_ids[r, :sl]
            real_mm = mm_token_type_ids[r, :sl] if mm_token_type_ids is not None else None
        enc = {'input_ids': real_ids[None]}
        n_img = int(mm_counts[r].get('image', 0))
        if image_grid_thw is not None and n_img > 0:
            grid_r = image_grid_thw[img_off:img_off + n_img]
            n_patch = int(grid_r.prod(-1).sum())
            if pixel_values is not None:
                enc['pixel_values'] = pixel_values[patch_off:patch_off + n_patch]
            enc['image_grid_thw'] = grid_r
            img_off += n_img
            patch_off += n_patch
        if real_mm is not None:
            enc['mm_token_type_ids'] = real_mm[None]

        nq = counts[r]
        real_list = real_ids.tolist()
        per_q_spans = []
        for qi in range(nq):
            markers = option_markers[q_cursor + qi]
            opens = [int(o) for o, _ in markers]
            closes = [int(c) for _, c in markers]
            per_q_spans.append((opens, closes))
        # ONE prefix pass per record; official shares it across the record's questions, cutting at the
        # MINIMUM (opens[0]-1) so every question keeps the token before its first option (where z_q is
        # read). rows_from_ids rebuilds each row as ids[L_use:opens[0]] (the instruction tail) + one option
        # block, and prefix_inputs cuts ids[:L_use] (the shared state) -- together they cover the whole
        # sequence with no gap.
        L_use = max(1, min(opens[0] - 1 for opens, _ in per_q_spans))
        penc = prefix_inputs(enc, L_use)
        ppos = rope_positions(rope_owner, penc)
        qrows = [rows_from_ids(real_list, L_use, opens, closes, k=len(opens)) for opens, closes in per_q_spans]
        with _grad_ckpt_disabled(backbone_model):
            outs = branch_questions(
                origin_forward, penc, qrows, dev, open_ids, close_ids, lm_head=lm_head, prefix_pos=ppos,
                pad_id=pad_id)
        per_question_features.extend(outs)
        q_cursor += nq

    return head.forward_features(per_question_features, meta)


class OmniJevModelLoader(ScoringModelLoader, Qwen3_5Loader):
    """OmniJev loader: load the hybrid Qwen3.5-4B base, register the option tokens, attach the fp32
    OmniJevHead, and patch forward with the two-stage cached BRANCH (decision G1).

    MRO is OmniJevModelLoader -> ScoringModelLoader -> Qwen3_5Loader -> ... -> ModelLoader. Unlike JEV /
    Clef (which let ScoringModelLoader.get_model do base -> build_head -> attach), OmniJev must register
    the option tokens and restore their embeddings on the BASE BEFORE swift's PEFT path loads the
    adapter-only checkpoint (official order: add_option_tokens(base) then PeftModel.from_pretrained(base,
    ckpt)); resize_token_embeddings after an adapter is attached would desync it. So get_model is
    overridden to call Qwen3_5Loader.get_model directly (skipping ScoringModelLoader's), add the option
    tokens, then build + attach the head. attach_scoring_head is overridden to patch the branch forward
    instead of the single-pass _scoring_head_forward.

    RUN-PHASE VERIFY (no local oracle): the adapter-only snapshot ships no config.json, so is_hybrid is
    asserted from the live base; the resize-then-PEFT ordering, the transformers 5.x cache .layers assumed
    by expand_cache, get_rope_index's signature, and DDP's reducer under several forwards per step are all
    checked against a real Qwen3.5-4B in the run phase, not during authoring."""
    head_cls = OmniJevHead

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        model = Qwen3_5Loader.get_model(self, model_dir, config, processor, model_kwargs)
        if not is_hybrid(model):
            raise ValueError(
                'OmniJev requires the hybrid (gated-deltanet linear-attention) Qwen3.5-4B backbone: on a '
                'linear-attention model a block-diagonal mask cannot isolate options, so the branch '
                'forward is mandatory. The loaded base reports no linear-attention layers '
                '(config.text_config.layer_types); point --model at the OmniJev base checkpoint.')
        # The official OmniJev release is adapter-only with `modules_to_save=None`, so the two option
        # tokens' embeddings are NOT carried by the LoRA -- `new_tok_emb.pt` (shipped beside head.pt /
        # ord.pt in the adapter dir) is their only source, and it is REQUIRED whenever the official head
        # is loaded (OmniJevHead.from_pretrained always strict-loads head.pt). Resolve it exactly like
        # the head artifacts (env override OMNIJEV_NEW_TOK_EMB, else model_dir) so a split base/adapter
        # layout (`--model Qwen/Qwen3.5-4B --adapters tinnel123/OmniJev` + OMNIJEV_* envs) restores the
        # real embeddings instead of silently mean-initialising the very vectors the head reads at the
        # option markers (a mean-init fallback would corrupt inference numerics with no error).
        emb_path = OmniJevHead._locate(model_dir, 'new_tok_emb.pt', 'OMNIJEV_NEW_TOK_EMB')
        if emb_path is None:
            raise FileNotFoundError(
                f'OmniJev new_tok_emb.pt not found under {model_dir!r}. The option-token embeddings are '
                'not in the LoRA adapter (modules_to_save=None), so they must be restored from '
                'new_tok_emb.pt; set OMNIJEV_NEW_TOK_EMB to its path (it ships beside head.pt in the '
                'tinnel123/OmniJev adapter dir).')
        add_option_tokens(model, processor, emb_path)
        tokenizer = processor.tokenizer
        self._omnijev_open_ids = tokenizer(OPT_OPEN, add_special_tokens=False)['input_ids']
        self._omnijev_close_ids = tokenizer(OPT_CLOSE, add_special_tokens=False)['input_ids']
        self._omnijev_tokenizer = tokenizer
        head = self.build_head(model, model_dir, config)
        head.set_marker_lengths(len(self._omnijev_open_ids), len(self._omnijev_close_ids))
        self.attach_scoring_head(model, head)
        return model

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        head = self._match_head_dtype_device(model, head)
        model.scoring_head = head
        if head.needs_output_embeddings:
            head.set_output_embeddings(model.get_output_embeddings())
        for p in head.parameters():
            p.requires_grad_(True)
        open_ids = getattr(self, '_omnijev_open_ids', None)
        close_ids = getattr(self, '_omnijev_close_ids', None)
        tokenizer = getattr(self, '_omnijev_tokenizer', None)
        if not open_ids or not close_ids or tokenizer is None:
            raise RuntimeError(
                'OmniJevModelLoader.attach_scoring_head needs the option tokens registered first; go '
                'through get_model (which runs add_option_tokens), not attach_scoring_head directly.')
        origin_forward = model.forward

        @wraps(origin_forward.__func__ if hasattr(origin_forward, '__func__') else origin_forward)
        def new_forward(self, *args, **kwargs):
            return _omnijev_branch_forward(
                self, *args, head=head, origin_forward=origin_forward, backbone_model=model,
                tokenizer=tokenizer, open_ids=open_ids, close_ids=close_ids, **kwargs)

        patch_module_forward(model, new_forward)


register_model(
    ModelMeta(
        MLLMModelType.omnijev,
        [ModelGroup([
            Model('tinnel123/OmniJev', 'tinnel123/OmniJev'),
        ], TemplateType.omnijev)],
        OmniJevModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        # `architectures` is intentionally EMPTY -- same reason as JEV/Clef: OmniJev's base is a
        # Qwen3_5ForConditionalGeneration, which MLLMModelType.qwen3_5 already claims for arch-based
        # auto-detection; re-declaring it would make an unnamed Qwen3.5 checkpoint resolve to multiple
        # model_types. OmniJev is matched by NAME (tinnel123/OmniJev) or an explicit --model_type omnijev.
        architectures=[],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'decision']))


class GenericDecisionLoader(ScoringModelLoader):
    """Train any CausalLM into a JEV-style decision model from scratch.

    Unlike ``JevModelLoader`` / ``ClefModelLoader`` / ``OmniJevModelLoader`` (which bind to
    ``Qwen3_5Loader`` via multiple inheritance and strict-load factory head weights from the model
    dir), this loader does NOT inherit from any model-specific loader. It uses the base
    ``ModelLoader.get_model`` which resolves ``AutoModelForCausalLM``, then builds a
    ``JevVerbalizerHead`` via ``from_tokenizer`` (random init, no factory weights) and attaches it.

    The user picks this loader with ``--task_type decision --template_type generic_decision``
    (or ``--model_type generic_decision``). The base CausalLM is any model that
    ``AutoModelForCausalLM.from_pretrained`` can load -- Qwen2.5, Llama, Mistral, etc.

    The LoRA adapter (with ``lm_head`` in ``target_modules``) and the head's 24 fp32 bias scalars
    are jointly trained (plan decision C). After training, the checkpoint carries:
    ``adapter_model.safetensors`` (LoRA) + ``scoring_head.safetensors`` (head bias) +
    ``scoring_head.json`` (verbalizer_ids / ranges / temperatures meta), and can be deployed with
    the standard swift inference path.
    """
    head_cls = JevVerbalizerHead

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        # Load the base via the standard ModelLoader path (AutoModelForCausalLM, not a
        # model-specific *ForConditionalGeneration). This is the key difference from
        # JevModelLoader which resolves to Qwen3_5Loader.get_model.
        model = ModelLoader.get_model(self, model_dir, config, processor, model_kwargs)
        # Build a randomly-initialized JevVerbalizerHead from the tokenizer.
        tokenizer = self._get_tokenizer(processor)
        head = self.head_cls.from_tokenizer(tokenizer, config=config)
        self.attach_scoring_head(model, head)
        return model

    @staticmethod
    def _get_tokenizer(processor: Processor):
        """Extract the tokenizer from a Processor (tokenizer, AutoProcessor, etc.)."""
        from transformers import PreTrainedTokenizerBase
        if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
            return processor.tokenizer
        return processor


register_model(
    ModelMeta(
        LLMModelType.generic_decision,
        [],  # Matched by --model_type or --task_type + --template_type, not by model id.
        GenericDecisionLoader,
        template=TemplateType.generic_decision,
        task_type='decision',
        architectures=[],  # Resolved from the base model's config; not declared here.
        tags=['decision']))
