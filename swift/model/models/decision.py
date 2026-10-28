# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import torch
from contextlib import contextmanager
from functools import wraps
from transformers import PretrainedConfig, PreTrainedModel
from typing import Dict, Optional

from swift.model.decision_head import (ClefJointSchemaHead, JevVerbalizerHead, KevPointerHead, LayaHead, OmniJevHead,
                                       ScoringHead, Thx02ReadoutHead, load_thx02_decision_config)
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
        if meta is None:
            raise ValueError("Decision forward requires decision_meta from the decision template's _encode "
                             '(kinds / n_options / model-specific positions); got none.')
        kwargs.pop('labels', None)
        kwargs['return_dict'] = True
        if head.encoder_backbone:
            output = origin_forward(*args, **kwargs)
            hidden_states = output.last_hidden_state
        else:
            kwargs['output_hidden_states'] = True
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


class KevModelLoader(ScoringModelLoader, Qwen3_5Loader):

    head_cls = KevPointerHead

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        # Adapter-only footgun: a kev repo passed directly to --model has no config.json / base weights.
        if not os.path.isfile(os.path.join(model_dir, 'config.json')):
            raise ValueError(f'{model_dir!r} is an adapter-only Kev checkpoint (no config.json, no base weights). Use '
                             '--model Qwen/Qwen3.5-{0.8,4,9}B-Base --model_type kev --adapters <kev adapter dir>, and '
                             'export KEV_HEAD=<adapter dir>/head.pt (see examples/train/decision/kev.sh).')
        model = super().get_model(model_dir, config, processor, model_kwargs)
        mapping = {r'^layers\.(?=\d)': r'model.language_model.layers.'}
        mapping.update(getattr(model, '_checkpoint_conversion_mapping', None) or {})
        model._checkpoint_conversion_mapping = mapping
        if not is_hybrid(model):
            raise ValueError('Kev requires the hybrid (gated-deltanet linear-attention) Qwen3.5 backbone '
                             '(config.text_config.layer_types); point --model at a Qwen3.5-*-Base '
                             'checkpoint.')
        return model


register_model(
    ModelMeta(
        MLLMModelType.kev, [
            ModelGroup([
                Model('jaredpalmer/kev-0.8b', 'jaredpalmer/kev-0.8b'),
                Model('jaredpalmer/kev-4b', 'jaredpalmer/kev-4b'),
                Model('jaredpalmer/kev-9b', 'jaredpalmer/kev-9b'),
            ], TemplateType.kev)
        ],
        KevModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        architectures=[],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['decision']))


def enable_thx02_noncausal_attention(language_model) -> None:
    """THX-02 reads every question at its own marker inside ONE sequence, so its full-attention layers
    attend bidirectionally over the non-padding tokens; the gated-deltanet (linear-attention) layers stay
    causal. Installed as a forward pre-hook on the text model, converting the 2D padding mask into the
    per-layer-type mask mapping that Qwen3.5 accepts."""
    from transformers.masking_utils import create_recurrent_attention_mask

    def mask_inputs(module, args, kwargs):
        if kwargs.get('past_key_values') is not None or kwargs.get('use_cache'):
            raise ValueError('THX-02 noncausal decision forward does not support a KV cache.')
        attention_mask = kwargs.get('attention_mask')
        if isinstance(attention_mask, dict):
            return args, kwargs
        embeddings = kwargs.get('inputs_embeds')
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs['input_ids'])
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        kwargs['attention_mask'] = {
            'full_attention':
            attention_mask[:, None, None, :].bool(),
            'linear_attention':
            create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=attention_mask),
        }
        return args, kwargs

    language_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


class Thx02ModelLoader(ScoringModelLoader, Qwen3_5Loader):
    """THX-02: a Qwen3.5 hybrid backbone (vision + gated-deltanet/full attention) with noncausal full
    attention and an inline readout head. The repo's top-level `config.json` is THX-02's own
    (`model_type: thx02`) and carries the backbone configuration under `backbone_config`; the backbone
    weights (`model.safetensors`) are stored without the `model.` prefix and without an lm_head."""
    head_cls = Thx02ReadoutHead

    def get_config(self, model_dir: str) -> PretrainedConfig:
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file, 'r') as f:
            top = json.load(f)
        if top.get('model_type') != 'thx02' or 'backbone_config' not in top:
            return super().get_config(model_dir)
        from transformers import Qwen3_5Config
        return Qwen3_5Config(**top['backbone_config'])

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        processor = super().get_processor(model_dir, config)
        decision_config = load_thx02_decision_config(model_dir)
        # Answer codes (A, B, ..., one token each) and the image pixel bounds THX-02 was trained with. These
        # bounds are the model's own; `--max_pixels` still applies first (the template rescales larger images
        # before the processor), so it can lower the resolution down to 65536 px but not raise it above the bound.
        processor.thx02_codes = list(decision_config['codes'])
        max_pixels = int(decision_config.get('max_pixels') or 262144)
        processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': max_pixels}
        return processor

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        model = Qwen3_5Loader.get_model(self, model_dir, config, processor, model_kwargs)
        enable_thx02_noncausal_attention(model.model.language_model)
        # The forward never applies lm_head. THX-02 ties it to embed_tokens (no extra memory, gets gradients
        # through the embeddings); should a config ever untie it, freeze the copy so full fine-tuning under DDP
        # does not see a parameter without gradients.
        lm_head, embeddings = model.get_output_embeddings(), model.get_input_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() != embeddings.weight.data_ptr():
            lm_head.weight.requires_grad_(False)
        head = self.build_head(model, model_dir, config)
        self.attach_scoring_head(model, head)
        from swift.model.decision_head import maybe_load_trained_scoring_head
        maybe_load_trained_scoring_head(model, model_dir)
        return model

    def attach_scoring_head(self, model: PreTrainedModel, head: ScoringHead) -> None:
        head = self._match_head_dtype_device(model, head)
        model.scoring_head = head
        origin_forward = model.forward

        @wraps(origin_forward.__func__ if hasattr(origin_forward, '__func__') else origin_forward)
        def new_forward(self, *args, **kwargs):
            return Thx02ModelLoader._thx02_forward(self, *args, head=head, **kwargs)

        patch_module_forward(model, new_forward)

    @staticmethod
    def _thx02_forward(self, *args, head: ScoringHead, **kwargs):
        meta = kwargs.pop('decision_meta', None)
        if meta is None:
            raise ValueError('THX-02 forward requires decision_meta from Thx02Template._encode.')
        for key in ('labels', 'logits_to_keep', 'output_hidden_states', 'return_dict', 'text_position_ids'):
            kwargs.pop(key, None)
        kwargs['use_cache'] = False
        # The readout reads the normalized last hidden state at each marker; the base model's lm_head is
        # never applied (THX-02 ships none), so call the Qwen3.5 backbone directly.
        output = self.model(*args, **kwargs)
        return head(output.last_hidden_state, None, meta)


register_model(
    ModelMeta(
        MLLMModelType.thx02, [ModelGroup([
            Model('HAL-X/THX-02', 'HAL-X-AI/THX-02'),
        ], TemplateType.thx02)],
        Thx02ModelLoader,
        model_arch=ModelArch.qwen2_vl,
        task_type='decision',
        architectures=['THXDecisionModel'],
        requires=['transformers>=5.13.0', 'qwen_vl_utils>=0.0.14'],
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


register_model(
    ModelMeta(
        LLMModelType.generic_decision, [],
        GenericDecisionLoader,
        template=TemplateType.generic_decision,
        task_type='decision',
        architectures=[],
        tags=['decision']))


class LayaModelLoader(ScoringModelLoader):

    head_cls = LayaHead

    @staticmethod
    def _resolve_checkpoint(model_dir: str) -> str:
        candidates = [
            os.environ.get('LAYA_CHECKPOINT'), model_dir,
            os.path.join(model_dir, 'typed-decisions'),
            os.path.join(model_dir, 'multilingual')
        ]
        for cand in candidates:
            if cand and (os.path.isfile(os.path.join(cand, 'model.safetensors'))
                         or os.path.isfile(os.path.join(cand, 'model.safetensors.index.json'))):
                return cand
        raise FileNotFoundError(
            f'Laya checkpoint not found under {model_dir!r} (checked the root and the typed-decisions / '
            'multilingual subfolders, plus the LAYA_CHECKPOINT env override). Expected model.safetensors '
            'with encoder.*-prefixed backbone weights; see convaiinnovations/laya on ModelScope.')

    @classmethod
    def _is_laya_stub_config(cls, path: str) -> bool:
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            return str(raw.get('model_type', '')).lower() == 'laya'
        except (json.JSONDecodeError, OSError):
            return False

    def get_config(self, model_dir: str) -> PretrainedConfig:
        from transformers import AutoConfig
        ckpt = self._resolve_checkpoint(model_dir)
        encoder_dir = os.path.join(ckpt, 'encoder')
        if os.path.isfile(os.path.join(encoder_dir, 'config.json')):
            config_dir = encoder_dir
        else:
            root_config = os.path.join(ckpt, 'config.json')
            if not os.path.isfile(root_config) or self._is_laya_stub_config(root_config):
                raise FileNotFoundError(
                    f'Laya encoder config not found: no encoder/config.json under {ckpt!r} and the root '
                    'config.json is the Hub-download stub.')
            config_dir = ckpt
        config = AutoConfig.from_pretrained(config_dir, trust_remote_code=self.default_trust_remote_code)
        # Same as ModernBertLoader: disable the ModernBERT reference compile path.
        config.reference_compile = False
        return config

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        from transformers import AutoTokenizer
        ckpt = self._resolve_checkpoint(model_dir)
        tok_dir = os.path.join(ckpt, 'tokenizer')
        if not os.path.isfile(os.path.join(tok_dir, 'tokenizer_config.json')):
            tok_dir = ckpt
        return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=self.default_trust_remote_code)

    @staticmethod
    def _load_checkpoint_weights(ckpt: str) -> Dict:
        from safetensors.torch import load_file
        index_path = os.path.join(ckpt, 'model.safetensors.index.json')
        weights: Dict = {}
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
            for shard in sorted(set(index.get('weight_map', {}).values())):
                weights.update(load_file(os.path.join(ckpt, shard)))
        else:
            weights = load_file(os.path.join(ckpt, 'model.safetensors'))
        if not weights:
            raise ValueError(f'Laya checkpoint {ckpt!r} has no weights.')
        return weights

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        from transformers import AutoModel
        ckpt = self._resolve_checkpoint(model_dir)
        weights = self._load_checkpoint_weights(ckpt)
        factory_layout = any(k.startswith('encoder.') for k in weights)
        if factory_layout:
            enc_sd = {k[len('encoder.'):]: v for k, v in weights.items() if k.startswith('encoder.')}
            head_sd = {
                k: v
                for k, v in weights.items() if not k.startswith('encoder.') and not k.startswith('act_head.')
            }
        else:
            enc_sd = {k: v for k, v in weights.items() if not k.startswith('scoring_head.')}
            head_sd = {k[len('scoring_head.'):]: v for k, v in weights.items() if k.startswith('scoring_head.')}
        if not enc_sd:
            raise ValueError(f'Laya checkpoint {ckpt!r} carries no backbone weights.')

        model = AutoModel.from_config(config)
        missing, unexpected = model.load_state_dict(enc_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f'Laya encoder strict load failed: missing={missing}, unexpected={unexpected}')
        if self.torch_dtype is not None:
            model = model.to(self.torch_dtype)

        head = self.build_head(model, ckpt, config)
        if head_sd:
            missing_h, unexpected_h = head.load_state_dict(head_sd, strict=False)
            if unexpected_h or missing_h:
                raise RuntimeError(f'Laya head strict load failed: missing={missing_h}, unexpected={unexpected_h}')
        self.attach_scoring_head(model, head)
        from swift.model.decision_head import maybe_load_trained_scoring_head
        maybe_load_trained_scoring_head(model, model_dir)
        return model


register_model(
    ModelMeta(
        LLMModelType.laya,
        [ModelGroup([
            Model('convaiinnovations/laya', 'convaiinnovations/laya'),
        ], TemplateType.laya)],
        LayaModelLoader,
        template=TemplateType.laya,
        task_type='decision',
        architectures=[],
        requires=['transformers>=4.48'],
        tags=['decision', 'bert']))
