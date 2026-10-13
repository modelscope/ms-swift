# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from functools import wraps
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional

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


class ScoringModelLoader(ModelLoader):

    head_cls: Optional[type] = None

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, config, processor, model_kwargs)
        head = self.build_head(model, model_dir, config)
        self.attach_scoring_head(model, head)
        from swift.model.decision_head import maybe_load_trained_scoring_head
        maybe_load_trained_scoring_head(model, model_dir)
        return model

    def build_head(self, model: PreTrainedModel, model_dir: str, config: PretrainedConfig) -> ScoringHead:
        if self.head_cls is None:
            raise NotImplementedError('ScoringModelLoader subclasses must set `head_cls` or override `build_head`.')
        return self.head_cls.from_pretrained(model_dir, model=model, config=config)

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        head = self._match_head_dtype_device(model, head)
        model.scoring_head = head
        if head.needs_output_embeddings:
            head.set_output_embeddings(model.get_output_embeddings())

        origin_forward = model.forward

        @wraps(origin_forward.__func__ if hasattr(origin_forward, '__func__') else origin_forward)
        def new_forward(self, *args, **kwargs):
            return ScoringModelLoader._scoring_head_forward(
                self, *args, head=head, origin_forward=origin_forward, **kwargs)

        patch_module_forward(model, new_forward)

    @staticmethod
    def _scoring_head_forward(self, *args, head: ScoringHead, origin_forward, **kwargs):
        meta = kwargs.pop('decision_meta', None)
        kwargs.pop('labels', None)
        kwargs['output_hidden_states'] = True
        kwargs['return_dict'] = True
        output = origin_forward(*args, **kwargs)
        hidden_states = output.hidden_states[-1]
        lm_logits = output.logits if head.needs_lm_logits else None
        return head(hidden_states, lm_logits, meta)

    @staticmethod
    def _match_head_dtype_device(model: PreTrainedModel, head: ScoringHead) -> ScoringHead:
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
    head_cls = JevVerbalizerHead


register_model(
    ModelMeta(
        MLLMModelType.jev, [ModelGroup([
            Model('autotrust/JEV-27B-VL', 'autotrust/JEV-27B-VL'),
        ], TemplateType.jev)],
        JevModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        architectures=[],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'decision']))


class ClefModelLoader(ScoringModelLoader, Qwen3_5Loader):
    head_cls = ClefJointSchemaHead


register_model(
    ModelMeta(
        MLLMModelType.clef, [ModelGroup([
            Model('Cloudflare/clef', 'Cloudflare/clef'),
        ], TemplateType.clef)],
        ClefModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        architectures=[],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'decision']))


class OmniJevModelLoader(ScoringModelLoader, Qwen3_5Loader):
    head_cls = OmniJevHead

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        model = Qwen3_5Loader.get_model(self, model_dir, config, processor, model_kwargs)
        if not is_hybrid(model):
            raise ValueError('OmniJev requires the hybrid (gated-deltanet linear-attention) Qwen3.5-4B backbone: on a '
                             'linear-attention model a block-diagonal mask cannot isolate options, so the branch '
                             'forward is mandatory. The loaded base reports no linear-attention layers '
                             '(config.text_config.layer_types); point --model at the OmniJev base checkpoint.')
        emb_path = OmniJevHead._locate(model_dir, 'new_tok_emb.pt', 'OMNIJEV_NEW_TOK_EMB')
        if emb_path is None:
            raise FileNotFoundError(
                f'OmniJev new_tok_emb.pt not found under {model_dir!r}. The option-token embeddings are '
                'not in the LoRA adapter (modules_to_save=None), so they must be restored from '
                'new_tok_emb.pt; set OMNIJEV_NEW_TOK_EMB to its path (it ships beside head.pt in the '
                'AI-ModelScope/OmniJev adapter dir).')
        add_option_tokens(model, processor, emb_path)
        tokenizer = processor.tokenizer
        self._omnijev_open_ids = tokenizer(OPT_OPEN, add_special_tokens=False)['input_ids']
        self._omnijev_close_ids = tokenizer(OPT_CLOSE, add_special_tokens=False)['input_ids']
        self._omnijev_tokenizer = tokenizer
        head = self.build_head(model, model_dir, config)
        head.set_marker_lengths(len(self._omnijev_open_ids), len(self._omnijev_close_ids))
        self.attach_scoring_head(model, head)
        from swift.model.decision_head import maybe_load_trained_scoring_head
        maybe_load_trained_scoring_head(model, model_dir)
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
            raise RuntimeError('OmniJevModelLoader.attach_scoring_head needs the option tokens registered first; go '
                               'through get_model (which runs add_option_tokens), not attach_scoring_head directly.')
        origin_forward = model.forward

        @wraps(origin_forward.__func__ if hasattr(origin_forward, '__func__') else origin_forward)
        def new_forward(self, *args, **kwargs):
            return OmniJevModelLoader._omnijev_branch_forward(
                self,
                *args,
                head=head,
                origin_forward=origin_forward,
                backbone_model=model,
                tokenizer=tokenizer,
                open_ids=open_ids,
                close_ids=close_ids,
                **kwargs)

        patch_module_forward(model, new_forward)

    @staticmethod
    @contextmanager
    def _grad_ckpt_disabled(backbone_model):
        touched = [m for m in backbone_model.modules() if getattr(m, 'gradient_checkpointing', False)]
        for m in touched:
            m.gradient_checkpointing = False
        try:
            yield
        finally:
            for m in touched:
                m.gradient_checkpointing = True

    @staticmethod
    def _omnijev_branch_forward(self, *args, head, origin_forward, backbone_model, tokenizer, open_ids, close_ids,
                                **kwargs):
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
            L_use = max(1, min(opens[0] - 1 for opens, _ in per_q_spans))
            penc = prefix_inputs(enc, L_use)
            ppos = rope_positions(rope_owner, penc)
            qrows = [rows_from_ids(real_list, L_use, opens, closes, k=len(opens)) for opens, closes in per_q_spans]
            with OmniJevModelLoader._grad_ckpt_disabled(backbone_model):
                outs = branch_questions(
                    origin_forward,
                    penc,
                    qrows,
                    dev,
                    open_ids,
                    close_ids,
                    lm_head=lm_head,
                    prefix_pos=ppos,
                    pad_id=pad_id)
            per_question_features.extend(outs)
            q_cursor += nq

        return head.forward_features(per_question_features, meta)


register_model(
    ModelMeta(
        MLLMModelType.omnijev,
        [ModelGroup([
            Model('AI-ModelScope/OmniJev', 'AI-ModelScope/OmniJev'),
        ], TemplateType.omnijev)],
        OmniJevModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        architectures=[],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'decision']))


class GenericDecisionLoader(ScoringModelLoader):
    head_cls = JevVerbalizerHead

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        model = ModelLoader.get_model(self, model_dir, config, processor, model_kwargs)
        tokenizer = self._get_tokenizer(processor)
        head = self.head_cls.from_tokenizer(tokenizer, config=config)
        self.attach_scoring_head(model, head)
        from swift.model.decision_head import maybe_load_trained_scoring_head
        maybe_load_trained_scoring_head(model, model_dir)
        return model

    @staticmethod
    def _get_tokenizer(processor: Processor):
        from transformers import PreTrainedTokenizerBase
        if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
            return processor.tokenizer
        return processor


register_model(
    ModelMeta(
        LLMModelType.generic_decision, [],
        GenericDecisionLoader,
        template=TemplateType.generic_decision,
        task_type='decision',
        architectures=[],
        tags=['decision']))
