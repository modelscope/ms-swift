# Copyright (c) ModelScope Contributors. All rights reserved.
# The decision (System-1 scoring) families: one backbone loader + one attachable ``ScoringHead``.

from __future__ import annotations
import os
from functools import wraps
from typing import Any, Dict, Tuple

import json
import torch
from transformers import PretrainedConfig, PreTrainedModel

from swift.dev.model.decision_branch import (
    OPT_CLOSE,
    OPT_OPEN,
    add_option_tokens,
    branch_questions,
    find_rope_owner,
    grad_ckpt_disabled,
    is_hybrid,
    kev_branch_enabled,
    kev_forward_branch,
    prefix_inputs,
    rope_positions,
    rows_from_ids,
)
from swift.dev.model.decision_head import (
    ClefJointSchemaHead,
    JevVerbalizerHead,
    KevPointerHead,
    LayaHead,
    OmniJevHead,
    ScoringHead,
    StartLuxLetterHead,
    Thx02ReadoutHead,
    load_thx02_decision_config,
    maybe_load_trained_scoring_head,
)
from swift.model.patcher import patch_module_forward  # v5's own patcher lacks get_original_forward
from .base import ModelLoader, register_model
from .qwen import Qwen3_5Loader


def _register_thx02_autoconfig() -> None:
    """Teach ``AutoConfig`` the ``thx02`` model_type (idempotent; module import time).

    THX-02's top-level ``config.json`` is its own (``model_type: thx02`` with the Qwen3.5 backbone
    nested under ``backbone_config``), which stock transformers cannot load. ``Thx02Loader.build_config``
    parses it itself, but twinkle's internal ``Template(self.tokenizer_id)`` (built in every optimizer
    group) runs a plain ``AutoConfig.from_pretrained`` on the model dir and would raise. Registering a
    ``Qwen3_5Config`` subclass that reads ``backbone_config`` keeps every ``AutoConfig`` consumer
    working; re-registering the same class is a no-op.
    """
    from transformers import AutoConfig, Qwen3_5Config

    class Thx02BackboneConfig(Qwen3_5Config):
        model_type = "thx02"

        def __init__(self, backbone_config=None, **kwargs):
            kwargs.pop("architectures", None)
            if backbone_config:
                merged = dict(backbone_config)
                merged.update(kwargs)
                kwargs = merged
            super().__init__(**kwargs)

    AutoConfig.register("thx02", Thx02BackboneConfig)


_register_thx02_autoconfig()


def get_original_forward(module):
    """The module's pre-patch forward (ported from the ms-swift patcher)."""
    if hasattr(module, "_old_forward"):  # device_map + decision
        return module._old_forward
    return module.forward


def _tokenizer_of(processor) -> Any:
    """The processor's tokenizer, or the processor itself when it already is one."""
    return getattr(processor, "tokenizer", processor)


class ScoringLoader(ModelLoader):
    """The shared construction rule: backbone first, then this family's :class:`ScoringHead`"""

    task_type = "decision"
    head_cls = None  # subclasses declare

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        model = super().build_model(model_dir, config, processor, **kwargs)
        head = self.build_head(model, model_dir, config, processor)
        self._attach_and_load(model, head, model_dir)
        return model

    def build_head(self, model: PreTrainedModel, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        if self.head_cls is None:
            raise NotImplementedError(f"{type(self).__name__} must set `head_cls` or override `build_head`.")
        return self.head_cls.from_pretrained(model_dir, model=model, config=config)

    def _attach_and_load(self, model: PreTrainedModel, head: ScoringHead, model_dir: str) -> None:
        """Shared loader tail: attach the head, then overlay any trained head checkpoint."""
        self.attach_scoring_head(model, head)
        maybe_load_trained_scoring_head(model, model_dir)

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        """Attach the head for the forward-layer readout dispatch -- no forward patching.

        The placement rules (dtype/device alignment, the dual ``task_head`` / ``scoring_head``
        names, output-embedding wiring) live on the decision task's spec
        (``twinkle.model.tasks.Decision``); this loader only supplies the head object.
        """
        from twinkle.model.tasks import get_task_spec_obj

        get_task_spec_obj("decision").attach_transformers(model, head)

    def _attach_task_forward(self, model: PreTrainedModel, dispatcher, dispatch: Dict[str, Any]) -> None:
        """Install a load-time custom task forward (the 'task_forward' readout kind).

        The branch-isolation / direct-backbone families drive the backbone themselves
        (multiple passes over a shared KV prefix, or reading past the top-level lm_head),
        which the generic head readout cannot express. The custom forward replaces the
        module forward and is marked ``task_forward`` so twinkle's dispatch takes the
        plain call and lets it consume the batch's decision meta itself.
        """
        origin_forward = get_original_forward(model)

        @wraps(origin_forward.__func__ if hasattr(origin_forward, "__func__") else origin_forward)
        def new_forward(self, *args, **kwargs):
            return dispatcher(self, *args, **dispatch, **kwargs)

        patch_module_forward(model, new_forward)
        model.task_forward = new_forward

    def task_head_loader(self) -> Tuple[str, str]:
        """The ``(module, class)`` name pair a twinkle worker replays ``build_task_head`` on.

        A built head (an nn.Module with live tensors) cannot cross the Ray worker boundary, so
        the backend builders ship this NAME instead; the worker rebuilds the head from the
        local model path / config / tokenizer and attaches it (per-backend: to the HF model
        or to the last-PP chunk / a head slot on multi-LoRA).
        """
        return type(self).__module__, type(self).__name__

    @classmethod
    def build_task_head(cls, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        """Worker-side head build (classmethod: no loader instance on the worker).

        Backend-agnostic: called wherever the head must be built worker-side -- megatron's
        ``_attach_task_head`` / head slots, or a transformers multi-adapter attach. The default
        replays the family's ``head_cls.from_pretrained``; families whose head needs the
        tokenizer override. The transformers-only ``task_forward`` families override with a
        raise, and hidden-state heads fail at the megatron readout (``needs_lm_logits``).
        """
        if cls.head_cls is None:
            raise NotImplementedError(f"{cls.__name__} must set `head_cls` or override `build_task_head`.")
        return cls.head_cls.from_pretrained(model_dir, config=config)

    @staticmethod
    def _match_head_dtype_device(model: PreTrainedModel, head: ScoringHead) -> ScoringHead:
        # Thin alias: the alignment rule lives on the decision task's spec (single source);
        # the task_forward families' attach overrides still need it locally.
        from twinkle.model.tasks import get_task_spec_obj

        return get_task_spec_obj("decision").align_head(model, head)


@register_model
class JevLoader(ScoringLoader, Qwen3_5Loader):
    """JEV: Qwen3.5-27B-VL backbone + a verbalizer head reading option-token logits."""

    model_type = "jev"
    head_cls = JevVerbalizerHead
    models = ["autotrust/JEV-27B-VL"]
    # A decision family is a template/behaviour variant of its backbone, so it stays reachable by
    # id match or an explicit `--model_type` only: reverse lookup from the backbone's architecture
    # class name must keep landing on the one base family (qwen3_5).
    template = "jev"
    architectures = ()
    requires = ["transformers>=5.2.0", "qwen_vl_utils>=0.0.14", "decord"]
    tags = ["vision", "decision"]


@register_model
class ClefLoader(ScoringLoader, Qwen3_5Loader):
    """Clef: Qwen3.5 backbone + a joint schema head."""

    model_type = "clef"
    head_cls = ClefJointSchemaHead
    models = ["Cloudflare/clef"]
    template = "clef"
    architectures = ()
    requires = ["transformers>=5.2.0", "qwen_vl_utils>=0.0.14", "decord"]
    tags = ["vision", "decision"]


@register_model
class OmniJevLoader(ScoringLoader, Qwen3_5Loader):
    """OmniJev: hybrid Qwen3.5-4B backbone + option tokens + a branch-isolated head"""

    model_type = "omnijev"
    head_cls = OmniJevHead
    models = ["AI-ModelScope/OmniJev"]
    template = "omnijev"
    architectures = ()
    requires = ["transformers>=5.2.0", "qwen_vl_utils>=0.0.14", "decord"]
    tags = ["vision", "decision"]

    @staticmethod
    def _prepare_option_tokens(model, processor, model_dir: str) -> None:
        saved_ids = getattr(model.config, "omnijev_option_token_ids", None)
        if saved_ids is not None:
            tokenizer = processor.tokenizer
            tokens = [OPT_OPEN, OPT_CLOSE]
            vocab = tokenizer.get_vocab()
            actual_ids = [vocab.get(token) for token in tokens]
            vocab_size = model.get_input_embeddings().weight.shape[0]
            if (
                not isinstance(saved_ids, list)
                or len(saved_ids) != 2
                or actual_ids != saved_ids
                or any(type(idx) is not int or not 0 <= idx < vocab_size for idx in saved_ids)
                or saved_ids[0] == saved_ids[1]
                or any(
                    tokenizer(token, add_special_tokens=False)["input_ids"] != [idx]
                    for token, idx in zip(tokens, saved_ids)
                )
            ):
                raise ValueError("OmniJev checkpoint option-token IDs do not match its tokenizer and embeddings.")
            return
        emb_path = OmniJevHead._locate(model_dir, "new_tok_emb.pt", "OMNIJEV_NEW_TOK_EMB")
        if emb_path is None:
            raise FileNotFoundError(
                f"OmniJev new_tok_emb.pt not found under {model_dir!r}. The option-token embeddings are "
                "not in the LoRA adapter (modules_to_save=None), so they must be restored from "
                "new_tok_emb.pt; set OMNIJEV_NEW_TOK_EMB to its path (it ships beside head.pt in the "
                "AI-ModelScope/OmniJev adapter dir)."
            )
        model.config.omnijev_option_token_ids = add_option_tokens(model, processor, emb_path)

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        model = Qwen3_5Loader.build_model(self, model_dir, config, processor, **kwargs)
        if not is_hybrid(model):
            raise ValueError(
                "OmniJev requires the hybrid (gated-deltanet linear-attention) Qwen3.5-4B backbone: on a "
                "linear-attention model a block-diagonal mask cannot isolate options, so the branch "
                "forward is mandatory. The loaded base reports no linear-attention layers "
                "(config.text_config.layer_types); point --model at the OmniJev base checkpoint."
            )
        self._prepare_option_tokens(model, processor, model_dir)
        tokenizer = processor.tokenizer
        self._omnijev_open_ids = tokenizer(OPT_OPEN, add_special_tokens=False)["input_ids"]
        self._omnijev_close_ids = tokenizer(OPT_CLOSE, add_special_tokens=False)["input_ids"]
        self._omnijev_tokenizer = tokenizer
        head = self.build_head(model, model_dir, config, processor)
        self._attach_and_load(model, head, model_dir)
        return model

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        # The branch forward trains the head jointly with the adapter, so it must stay trainable
        # regardless of any later freeze sweep.
        for p in head.parameters():
            p.requires_grad_(True)
        head = self._match_head_dtype_device(model, head)
        model.scoring_head = head
        open_ids = getattr(self, "_omnijev_open_ids", None)
        close_ids = getattr(self, "_omnijev_close_ids", None)
        tokenizer = getattr(self, "_omnijev_tokenizer", None)
        if not open_ids or not close_ids or tokenizer is None:
            raise RuntimeError(
                "OmniJevLoader.attach_scoring_head needs the option tokens registered first; go "
                "through build_model (which runs add_option_tokens), not attach_scoring_head directly."
            )
        self._attach_task_forward(
            model,
            self._omnijev_branch_forward,
            {
                "head": head,
                "origin_forward": get_original_forward(model),
                "backbone_model": model,
                "tokenizer": tokenizer,
                "open_ids": open_ids,
                "close_ids": close_ids,
            },
        )

    @classmethod
    def build_task_head(cls, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        raise NotImplementedError(
            f"{cls.__name__} runs a load-time custom forward (option branch isolation over a shared "
            'HF KV cache) with no megatron equivalent; use the transformers backend for this family.'
        )

    def task_head_loader(self) -> Tuple[str, str]:
        # Driver-side twin of build_task_head's raise: fail at build time, before Ray starts.
        self.build_task_head("", None, None)

    @staticmethod
    def _omnijev_branch_forward(  # noqa: C901
        self, *args, head, origin_forward, backbone_model, tokenizer, open_ids, close_ids, **kwargs
    ):
        meta = kwargs.pop("decision_meta", None)
        kwargs.pop("labels", None)
        if meta is None:
            raise ValueError("OmniJev branch forward requires decision_meta from OmniJevTemplate._encode.")
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError(
                "OmniJev branch forward requires input_ids; OmniJevTemplate._post_encode must NOT convert the "
                "batch to inputs_embeds (the branch re-runs the backbone on the real token ids + pixels)."
            )
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        mm_token_type_ids = kwargs.get("mm_token_type_ids")
        dev = input_ids.device
        B, T = input_ids.shape[0], input_ids.shape[1]
        left_padding = bool(meta.get("left_padding", False))
        seq_lens = meta.get("seq_lens")
        counts = [int(c) for c in meta["counts"]]
        option_markers = meta["option_markers"]
        mm_counts = meta.get("mm_counts") or [{"image": 0, "video": 0}] * B
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
                real_ids = input_ids[r, T - sl :]
                real_mm = mm_token_type_ids[r, T - sl :] if mm_token_type_ids is not None else None
            else:
                real_ids = input_ids[r, :sl]
                real_mm = mm_token_type_ids[r, :sl] if mm_token_type_ids is not None else None
            enc = {"input_ids": real_ids[None]}
            n_img = int(mm_counts[r].get("image", 0))
            if image_grid_thw is not None and n_img > 0:
                grid_r = image_grid_thw[img_off : img_off + n_img]
                n_patch = int(grid_r.prod(-1).sum())
                if pixel_values is not None:
                    enc["pixel_values"] = pixel_values[patch_off : patch_off + n_patch]
                enc["image_grid_thw"] = grid_r
                img_off += n_img
                patch_off += n_patch
            if real_mm is not None:
                enc["mm_token_type_ids"] = real_mm[None]

            nq = counts[r]
            real_list = real_ids.tolist()
            per_q_spans = []
            for qi in range(nq):
                markers = option_markers[q_cursor + qi]
                opens = [int(o) for o, _ in markers]
                closes = [int(c) for _, c in markers]
                per_q_spans.append((opens, closes))
            L_use = max(1, min(opens[0] - 1 for opens, _ in per_q_spans))
            penc = prefix_inputs(enc, L_use)
            ppos = rope_positions(rope_owner, penc)
            qrows = [rows_from_ids(real_list, L_use, opens, closes, k=len(opens)) for opens, closes in per_q_spans]
            with grad_ckpt_disabled(backbone_model):
                outs = branch_questions(
                    origin_forward,
                    penc,
                    qrows,
                    dev,
                    open_ids,
                    close_ids,
                    lm_head=lm_head,
                    prefix_pos=ppos,
                    pad_id=pad_id,
                )
            per_question_features.extend(outs)
            q_cursor += nq

        return head.forward_features(per_question_features, meta)


@register_model
class KevLoader(ScoringLoader, Qwen3_5Loader):
    """Kev: hybrid Qwen3.5-Base text backbone + a pointer head."""

    model_type = "kev"
    head_cls = KevPointerHead
    models = ["jaredpalmer/kev-0.8b", "jaredpalmer/kev-4b", "jaredpalmer/kev-9b"]
    template = "kev"
    architectures = ()
    # Text-only Base checkpoints: not a vision family (legacy pinned the same via its tags).
    is_multimodal = False
    requires = ["transformers>=5.2.0", "qwen_vl_utils>=0.0.14", "decord"]
    tags = ["decision"]

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        # Adapter-only footgun: a kev repo passed directly to --model has no config.json / base weights.
        if not os.path.isfile(os.path.join(model_dir, "config.json")):
            raise ValueError(
                f"{model_dir!r} is an adapter-only Kev checkpoint (no config.json, no base weights). Use "
                "--model Qwen/Qwen3.5-{0.8,4,9}B-Base --model_type kev --adapters <kev adapter dir>, and "
                "export KEV_HEAD=<adapter dir>/head.pt (see examples/train/decision/kev.sh)."
            )
        model = Qwen3_5Loader.build_model(self, model_dir, config, processor, **kwargs)
        # Fail on a non-hybrid backbone BEFORE the head is attached, not after.
        if not is_hybrid(model):
            raise ValueError(
                "Kev requires the hybrid (gated-deltanet linear-attention) Qwen3.5 backbone "
                "(config.text_config.layer_types); point --model at a Qwen3.5-*-Base "
                "checkpoint."
            )
        mapping = {r"^layers\.(?=\d)": r"model.language_model.layers."}
        mapping.update(getattr(model, "_checkpoint_conversion_mapping", None) or {})
        model._checkpoint_conversion_mapping = mapping
        head = self.build_head(model, model_dir, config, processor)
        self._attach_and_load(model, head, model_dir)
        return model

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        head = self._match_head_dtype_device(model, head)
        model.scoring_head = head
        self._attach_task_forward(
            model,
            self._kev_forward,
            {"head": head, "origin_forward": get_original_forward(model), "backbone_model": model},
        )

    @classmethod
    def build_task_head(cls, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        raise NotImplementedError(
            f"{cls.__name__} runs a load-time custom forward (option branch isolation over a shared "
            'HF KV cache) with no megatron equivalent; use the transformers backend for this family.'
        )

    def task_head_loader(self) -> Tuple[str, str]:
        # Driver-side twin of build_task_head's raise: fail at build time, before Ray starts.
        self.build_task_head("", None, None)

    @staticmethod
    def _kev_forward(self, *args, head: ScoringHead, origin_forward, backbone_model, **kwargs):
        meta = kwargs.pop("decision_meta", None)
        if meta is None:
            raise ValueError(
                "Decision forward requires decision_meta from KevTemplate._encode "
                "(kinds / n_options / decide_idx / opt_idx); got none."
            )
        kwargs.pop("labels", None)
        kwargs["return_dict"] = True
        if kev_branch_enabled():
            out = kev_forward_branch(origin_forward, head, backbone_model, kwargs, meta)
            if out is not None:
                return out
        kwargs["output_hidden_states"] = True
        output = origin_forward(*args, **kwargs)
        hidden_states = output.hidden_states[-1]
        return head(hidden_states, None, meta)


def enable_thx02_noncausal_attention(language_model) -> None:
    """THX-02 reads every question at its own marker inside ONE sequence, so its full-attention layers
    attend bidirectionally over the non-padding tokens; the gated-deltanet (linear-attention) layers stay
    causal. Installed as a forward pre-hook on the text model, converting the 2D padding mask into the
    per-layer-type mask mapping that Qwen3.5 accepts."""
    from transformers.masking_utils import create_recurrent_attention_mask

    def mask_inputs(module, args, kwargs):
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("THX-02 noncausal decision forward does not support a KV cache.")
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, dict):
            return args, kwargs
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        kwargs["attention_mask"] = {
            "full_attention": attention_mask[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=attention_mask
            ),
        }
        return args, kwargs

    language_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


@register_model
class Thx02Loader(ScoringLoader, Qwen3_5Loader):
    """THX-02: a Qwen3.5 hybrid backbone (vision + gated-deltanet/full attention) with noncausal full
    attention and an inline readout head. The repo's top-level `config.json` is THX-02's own
    (`model_type: thx02`) and carries the backbone configuration under `backbone_config`; the backbone
    weights (`model.safetensors`) are stored without the `model.` prefix and without an lm_head."""

    model_type = "thx02"
    head_cls = Thx02ReadoutHead
    models = [("HAL-X/THX-02", "HAL-X-AI/THX-02")]
    template = "thx02"
    architectures = ("THXDecisionModel",)
    requires = ["transformers>=5.13.0", "qwen_vl_utils>=0.0.14"]
    tags = ["vision", "decision"]

    def build_config(self, model_dir: str, **kwargs) -> PretrainedConfig:
        config_file = os.path.join(model_dir, "config.json")
        with open(config_file, "r") as f:
            top = json.load(f)
        if top.get("model_type") != "thx02" or "backbone_config" not in top:
            raise ValueError(
                f"{model_dir!r} is not a THX-02 checkpoint: its config.json must declare "
                "`model_type: thx02` with the Qwen3.5 backbone under `backbone_config`. "
                "thx02 has no build-on-vanilla-backbone flow (unlike kev); point --model at "
                "the THX-02 repo or a THX-02-format directory."
            )
        from transformers import Qwen3_5Config

        return Qwen3_5Config(**top["backbone_config"])

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        processor = super().build_processor(model_dir, config, **kwargs)
        decision_config = load_thx02_decision_config(model_dir)
        # Preserve the effective processor settings (including an environment override)
        # in full checkpoints, which do not carry the factory decision_config.json.
        config.thx02_decision_config = decision_config
        processor.thx02_codes = list(decision_config["codes"])
        max_pixels = int(decision_config.get("max_pixels") or 262144)
        processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": max_pixels}
        return processor

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        model = Qwen3_5Loader.build_model(self, model_dir, config, processor, **kwargs)
        enable_thx02_noncausal_attention(model.model.language_model)
        lm_head, embeddings = model.get_output_embeddings(), model.get_input_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() != embeddings.weight.data_ptr():
            lm_head.weight.requires_grad_(False)
        head = self.build_head(model, model_dir, config, processor)
        self._attach_and_load(model, head, model_dir)
        return model

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        # THX-02's dispatcher calls the backbone (self.model) directly -- past the top-level
        # lm_head the base model never applies -- so it is a task_forward, not a task_head.
        head = self._match_head_dtype_device(model, head)
        model.scoring_head = head
        self._attach_task_forward(model, self._thx02_forward, {"head": head})

    @classmethod
    def build_task_head(cls, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        raise NotImplementedError(
            f"{cls.__name__} runs a load-time custom forward (direct backbone readout past the "
            'top-level lm_head, plus noncausal attention pre-hooks) with no megatron equivalent; '
            'use the transformers backend for this family.'
        )

    def task_head_loader(self) -> Tuple[str, str]:
        # Driver-side twin of build_task_head's raise: fail at build time, before Ray starts.
        self.build_task_head("", None, None)

    @staticmethod
    def _thx02_forward(self, *args, head: ScoringHead, **kwargs):
        meta = kwargs.pop("decision_meta", None)
        if meta is None:
            raise ValueError("THX-02 forward requires decision_meta from Thx02Template._encode.")
        for key in ("labels", "logits_to_keep", "output_hidden_states", "return_dict", "text_position_ids"):
            kwargs.pop(key, None)
        kwargs["use_cache"] = False
        # The readout reads the normalized last hidden state at each marker; the base model's lm_head is
        # never applied (THX-02 ships none), so call the Qwen3.5 backbone directly.
        output = self.model(*args, **kwargs)
        return head(output.last_hidden_state, None, meta)


@register_model
class StartLuxLoader(ScoringLoader, Qwen3_5Loader):
    """StartLux: Qwen3.5 backbone + a letter head that needs the tokenizer at build time."""

    model_type = "startlux"
    head_cls = StartLuxLetterHead
    models = [
        ("StartLuxAI/StartLux-Decision-v1.1-4B-AutoThink", "startlux-models/StartLux-Decision-v1.1-4B-AutoThink"),
        ("StartLuxAI/StartLux-Decision-v1.1-9B-AutoThink", "startlux-models/StartLux-Decision-v1.1-9B-AutoThink"),
        ("StartLuxAI/StartLux-Decision-v1.1-27B-AutoThink", "startlux-models/StartLux-Decision-v1.1-27B-AutoThink"),
    ]
    template = "startlux"
    architectures = ()
    requires = ["transformers>=5.2.0", "qwen_vl_utils>=0.0.14", "decord"]
    tags = ["vision", "decision"]

    def build_head(self, model: PreTrainedModel, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        return self.head_cls.from_pretrained(model_dir, model=model, config=config, tokenizer=_tokenizer_of(processor))

    @classmethod
    def build_task_head(cls, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        return cls.head_cls.from_pretrained(model_dir, config=config, tokenizer=_tokenizer_of(processor))


@register_model
class GenericDecisionLoader(ScoringLoader):
    """Any causal-LM backbone + a verbalizer head sized from its tokenizer"""

    model_type = "generic_decision"
    head_cls = JevVerbalizerHead
    models = ()
    template = "generic_decision"
    architectures = ()
    model_cls = "transformers:AutoModelForCausalLM"
    tags = ["decision"]

    def build_head(self, model: PreTrainedModel, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        return self.head_cls.from_tokenizer(_tokenizer_of(processor), config=config)

    @classmethod
    def build_task_head(cls, model_dir: str, config: PretrainedConfig, processor) -> ScoringHead:
        return cls.head_cls.from_tokenizer(_tokenizer_of(processor), config=config)


@register_model
class LayaLoader(ScoringLoader):
    """Laya: a ModernBERT encoder backbone + a classification head, loaded from raw safetensors"""

    model_type = "laya"
    head_cls = LayaHead
    models = ["convaiinnovations/laya"]
    template = "laya"
    architectures = ()
    model_cls = "transformers:AutoModel"
    trust_remote_code = True
    requires = ["transformers>=4.48"]
    tags = ["decision", "bert"]

    @staticmethod
    def _resolve_checkpoint(model_dir: str) -> str:
        candidates = [
            os.environ.get("LAYA_CHECKPOINT"),
            model_dir,
            os.path.join(model_dir, "typed-decisions"),
            os.path.join(model_dir, "multilingual"),
        ]
        for cand in candidates:
            if cand and (
                os.path.isfile(os.path.join(cand, "model.safetensors"))
                or os.path.isfile(os.path.join(cand, "model.safetensors.index.json"))
            ):
                return cand
        raise FileNotFoundError(
            f"Laya checkpoint not found under {model_dir!r} (checked the root and the typed-decisions / "
            "multilingual subfolders, plus the LAYA_CHECKPOINT env override). Expected model.safetensors "
            "with encoder.*-prefixed backbone weights; see convaiinnovations/laya on ModelScope."
        )

    @classmethod
    def _is_laya_stub_config(cls, path: str) -> bool:
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            return str(raw.get("model_type", "")).lower() == "laya"
        except (json.JSONDecodeError, OSError):
            return False

    def build_config(self, model_dir: str, **kwargs) -> PretrainedConfig:
        self._apply_trust_remote_code(kwargs)
        from transformers import AutoConfig

        ckpt = self._resolve_checkpoint(model_dir)
        encoder_dir = os.path.join(ckpt, "encoder")
        if os.path.isfile(os.path.join(encoder_dir, "config.json")):
            config_dir = encoder_dir
        else:
            root_config = os.path.join(ckpt, "config.json")
            if not os.path.isfile(root_config) or self._is_laya_stub_config(root_config):
                raise FileNotFoundError(
                    f"Laya encoder config not found: no encoder/config.json under {ckpt!r} and the root "
                    "config.json is the Hub-download stub."
                )
            config_dir = ckpt
        kwargs.pop("revision", None)  # config_dir is a local subfolder, not a hub id
        config = AutoConfig.from_pretrained(config_dir, **kwargs)
        # Same as ModernBertLoader: disable the ModernBERT reference compile path.
        config.reference_compile = False
        return config

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        self._apply_trust_remote_code(kwargs)
        from transformers import AutoTokenizer

        ckpt = self._resolve_checkpoint(model_dir)
        tok_dir = os.path.join(ckpt, "tokenizer")
        if not os.path.isfile(os.path.join(tok_dir, "tokenizer_config.json")):
            tok_dir = ckpt
        return AutoTokenizer.from_pretrained(tok_dir, **kwargs)

    @staticmethod
    def _load_checkpoint_weights(ckpt: str) -> Dict:
        from safetensors.torch import load_file

        index_path = os.path.join(ckpt, "model.safetensors.index.json")
        weights: Dict = {}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            for shard in sorted(set(index.get("weight_map", {}).values())):
                weights.update(load_file(os.path.join(ckpt, shard)))
        else:
            weights = load_file(os.path.join(ckpt, "model.safetensors"))
        if not weights:
            raise ValueError(f"Laya checkpoint {ckpt!r} has no weights.")
        return weights

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        from transformers import AutoModel

        ckpt = self._resolve_checkpoint(model_dir)
        weights = self._load_checkpoint_weights(ckpt)
        factory_layout = any(k.startswith("encoder.") for k in weights)
        if factory_layout:
            enc_sd = {k[len("encoder.") :]: v for k, v in weights.items() if k.startswith("encoder.")}
            head_sd = {
                k: v for k, v in weights.items() if not k.startswith("encoder.") and not k.startswith("act_head.")
            }
        else:
            enc_sd = {k: v for k, v in weights.items() if not k.startswith("scoring_head.")}
            head_sd = {k[len("scoring_head.") :]: v for k, v in weights.items() if k.startswith("scoring_head.")}
        if not enc_sd:
            raise ValueError(f"Laya checkpoint {ckpt!r} carries no backbone weights.")

        model = AutoModel.from_config(config)
        missing, unexpected = model.load_state_dict(enc_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Laya encoder strict load failed: missing={missing}, unexpected={unexpected}")
        torch_dtype = self.model_info.torch_dtype
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        head = self.build_head(model, ckpt, config, processor)
        if head_sd:
            missing_h, unexpected_h = head.load_state_dict(head_sd, strict=False)
            if unexpected_h or missing_h:
                raise RuntimeError(f"Laya head strict load failed: missing={missing_h}, unexpected={unexpected_h}")
        self._attach_and_load(model, head, model_dir)
        return model
