# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for assorted transformers-native multimodal families (from ``swift/model/models/mllm.py``).

Migrated: the transformers-native ``idefics3`` and ``pixtral``, plus the remote-code ``keye_vl`` /
``molmo`` / ``molmo2`` / ``dots_ocr`` (loaded via an Auto class + ``trust_remote_code`` -- the base g7
flag). Their legacy loaders only did device_map work (``get_class_from_dynamic_module`` to set
``_no_split_modules``) and output-clone / ``keye_vl_utils`` patches, all obsolete per PATCH_INVENTORY,
so they are dropped. The ``qwen2_gme`` embedding variant is a pure Qwen2-VL template variant in
``qwen.py``.

Not migrated here (see MODEL_MIGRATION.md):
  * ``keye_vl_1_5`` -- pinned ``transformers==4.52.4`` (dead on dev's 5.5).
  * ``sail_vl2`` -- pinned ``transformers<=4.51.3`` (dead on dev's 5.5). Its only loader logic is
    ``use_submodel_func(language_model)``, which ``delegate_to_submodel`` would now cover, but the
    version pin kills it regardless.
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class Idefics3Loader(ModelLoader):
    model_type = 'idefics3'
    model_cls = 'transformers:AutoModelForVision2Seq'
    architectures = ['Idefics3ForConditionalGeneration']
    template = 'idefics3'
    requires = ['transformers>=4.45']
    tags = ['vision']
    is_multimodal = True
    models = [('AI-ModelScope/Idefics3-8B-Llama3', 'HuggingFaceM4/Idefics3-8B-Llama3')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model='model.text_model',
            aligner='model.connector',
            vision_tower='model.vision_model',
        )


@register_model
class PixtralLoader(ModelLoader):
    model_type = 'pixtral'
    model_cls = 'transformers:LlavaForConditionalGeneration'
    architectures = ['LlavaForConditionalGeneration']
    template = 'pixtral'
    requires = ['transformers>=4.45']
    tags = ['vision']
    is_multimodal = True
    models = [('AI-ModelScope/pixtral-12b', 'mistral-community/pixtral-12b')]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llava_hf`, transformers>=4.52 (model.* prefix) branch.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )


@register_model
class KeyeVLLoader(ModelLoader):
    """Keye-VL: remote-code ``KeyeForConditionalGeneration``. Legacy patched
    ``keye_vl_utils.vision_process`` in the processor -- obsolete per PATCH_INVENTORY, dropped."""

    model_type = 'keye_vl'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['KeyeForConditionalGeneration']
    template = 'keye_vl'
    requires = ['keye_vl_utils']
    tags = ['vision']
    is_multimodal = True
    models = [('Kwai-Keye/Keye-VL-8B-Preview', 'Kwai-Keye/Keye-VL-8B-Preview')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model=['model', 'lm_head'], aligner='mlp_AR', vision_tower='visual')


@register_model
class MolmoLoader(ModelLoader):
    """Molmo: remote-code ``MolmoForCausalLM``. Legacy set ``_no_split_modules`` (device_map, obsolete)
    and cloned the embedding output (obsolete); both dropped."""

    model_type = 'molmo'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['MolmoForCausalLM']
    template = 'molmo'
    requires = ['transformers>=4.45']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('LLM-Research/Molmo-7B-O-0924', 'allenai/Molmo-7B-O-0924'),
        ('LLM-Research/Molmo-7B-D-0924', 'allenai/Molmo-7B-D-0924'),
        ('LLM-Research/Molmo-72B-0924', 'allenai/Molmo-72B-0924'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model='model.transformer',
            aligner='model.vision_backbone.image_projector',
            vision_tower='model.vision_backbone',
        )


@register_model
class Molmo2Loader(MolmoLoader):
    """Molmo2 shares the molmo ``model_arch`` but loads via AutoModelForImageTextToText."""

    model_type = 'molmo2'
    model_cls = 'transformers:AutoModelForImageTextToText'
    architectures = ['Molmo2ForConditionalGeneration']
    template = 'molmo2'
    requires = ['transformers>=4.57.1,<5', 'decord']
    tags = ['vision', 'video']
    models = [
        ('allenai/Molmo2-4B', 'allenai/Molmo2-4B'),
        ('allenai/Molmo2-8B', 'allenai/Molmo2-8B'),
        ('allenai/Molmo2-O-7B', 'allenai/Molmo2-O-7B'),
    ]


@register_model
class FlorenceLoader(ModelLoader):
    """Florence-2: remote-code ``Florence2ForConditionalGeneration``, a wrapper whose real decoder is
    ``language_model`` (a BART-style seq2seq), so ``process_model`` proxies ``forward``/``generate``
    there -- legacy's ``use_submodel_func(model, 'language_model', ['generate', 'forward'])``.

    Three legacy behaviours are real and kept:
      * ``ignore_check_imports`` around the load: the checkpoint's ``modeling_*.py`` declares imports
        (flash-attn among them) that are never reached on our paths, and the dependency check would
        otherwise refuse the dynamic import outright.
      * ``config.vision_config.model_type = 'davit'`` in ``process_config``: the shipped config leaves
        it unset, which breaks merge-lora (legacy's own comment) because the sub-config cannot be
        re-instantiated from a saved dict without a model_type.
      * ``vision_tower.enable_checkpoint = True``: the DaViT tower's own gradient-checkpointing switch,
        which transformers' generic ``gradient_checkpointing_enable`` does not reach.

    Dropped: the ``device_map == 'auto'`` override (dev does not use HF device_map; twinkle owns
    placement, per PATCH_INVENTORY.md).
    """

    model_type = 'florence'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['Florence2ForConditionalGeneration']
    template = 'florence'
    tags = ['vision']
    is_multimodal = True
    models = [
        ('AI-ModelScope/Florence-2-base-ft', 'microsoft/Florence-2-base-ft'),
        ('AI-ModelScope/Florence-2-base', 'microsoft/Florence-2-base'),
        ('AI-ModelScope/Florence-2-large', 'microsoft/Florence-2-large'),
        ('AI-ModelScope/Florence-2-large-ft', 'microsoft/Florence-2-large-ft'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.florence`: no aligner partition declared.
        return ModelArch(language_model=['language_model'], vision_tower=['vision_tower'])

    def process_config(self, config):
        config.vision_config.model_type = 'davit'  # fix merge-lora
        return config

    def build_model(self, model_dir: str, config, processor, **kwargs):
        with self.ignore_check_imports():
            return super().build_model(model_dir, config, processor, **kwargs)

    def process_model(self, model):
        model.vision_tower.enable_checkpoint = True
        self.delegate_to_submodel(model, 'language_model', ['generate', 'forward'])
        return model


@register_model
class MolmoELoader(MolmoLoader):
    """MolmoE-1B: the MoE sibling of Molmo, whose remote-code class name is ``OLMoForCausalLM`` (not
    ``MolmoForCausalLM``), so it owns that architecture name for reverse-lookup. Shares the ``molmo``
    template and module layout.

    Two legacy behaviours are real and kept:
      * **fp32 weights.** Legacy declared ``torch_dtype=torch.float32`` on the ModelMeta: this
        checkpoint is numerically unstable in bf16/fp16. dev has no family-level default-dtype
        declaration, so it is injected as a ``setdefault`` in ``build_model`` -- an explicit user
        ``--torch_dtype`` still wins.
      * **``config.to_dict`` repair.** ``vision_backbone`` is a plain object on the config rather than
        a nested ``PretrainedConfig``, so stock ``to_dict()`` silently omits it and any
        save/serialize round-trip loses the vision settings. The wrapper folds it back in (and drops
        the two bookkeeping keys it would otherwise leak).

    Dropped: ``patch_output_clone`` on the token embedding (obsolete per PATCH_INVENTORY -- a
    reentrant-gradient-checkpointing workaround).
    """

    model_type = 'molmoe'
    architectures = ['OLMoForCausalLM']
    template = 'molmo'
    requires = ['transformers>=4.45']
    tags = ['vision']
    is_moe = True
    models = [('LLM-Research/MolmoE-1B-0924', 'allenai/MolmoE-1B-0924')]

    def build_model(self, model_dir: str, config, processor, **kwargs):
        import torch
        kwargs.setdefault('dtype', torch.float32)
        return super().build_model(model_dir, config, processor, **kwargs)

    def process_model(self, model):
        from types import MethodType
        config = model.config
        if hasattr(config, '_to_dict'):
            return model

        def to_dict(self, *args, **kwargs):
            res = self._to_dict(*args, **kwargs)
            res['vision_backbone'] = self.vision_backbone.__dict__
            res.pop('to_dict', None)
            res.pop('_to_dict', None)
            return res

        config._to_dict = config.to_dict
        config.to_dict = MethodType(to_dict, config)
        return model


@register_model
class DotsOCRLoader(ModelLoader):
    """dots.ocr: remote-code ``DotsOCRForCausalLM``. Legacy only set the vision ``_no_split_modules``
    (device_map, obsolete), dropped."""

    model_type = 'dots_ocr'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['DotsOCRForCausalLM']
    template = 'dots_ocr'
    requires = ['transformers>=4.51.0']
    is_multimodal = True
    models = [('rednote-hilab/dots.ocr', 'rednote-hilab/dots.ocr')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model='model')


@register_model
class MegrezOmniLoader(ModelLoader):
    """Megrez-3B-Omni: remote-code ``MegrezO``, a thin wrapper whose real language model hangs off
    ``model.llm`` -- so ``process_model`` proxies the wrapper's forward/generate/get_input_embeddings
    to it via ``delegate_to_submodel`` (legacy ``use_submodel_func(model, 'llm')``; the device_map
    ``_no_split_modules`` and ``patch_output_clone`` around it are obsolete per PATCH_INVENTORY and
    dropped).

    Processor-from-model coupling: Megrez builds its processor *from the loaded model*
    (``model._get_or_init_processor()``), the reverse of dev's usual config->processor->model order.
    ``build_processor`` expresses this by instantiating the model to obtain its processor. (The dev
    loader hooks are registry-only today; if the orchestrator later hands the built model to a
    processor hook, this can drop the extra instantiation.)
    """

    model_type = 'megrez_omni'
    model_cls = 'transformers:AutoModel'
    trust_remote_code = True
    architectures = ['MegrezO']
    template = 'megrez_omni'
    tags = ['vision', 'audio']
    is_multimodal = True
    models = [('InfiniAI/Megrez-3B-Omni', 'Infinigence/Megrez-3B-Omni')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model=['llm'], vision_tower=['vision', 'audio'])

    def build_processor(self, model_dir: str, config, **kwargs):
        kwargs.setdefault('trust_remote_code', True)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_dir, config=config, **kwargs)
        return model._get_or_init_processor()

    def process_model(self, model):
        self.delegate_to_submodel(model, 'llm')
        return model


@register_model
class JinaRerankerM0Loader(ModelLoader):
    """jina-reranker-m0: a multimodal reranker. Remote-code, loaded via ``AutoModel`` +
    ``trust_remote_code`` so the checkpoint's own head is used. ``task_type='reranker'``.

    ``process_model`` wraps ``forward`` to emit a ``SequenceClassifierOutputWithPast`` whose
    ``logits`` are the model's scalar score minus a fixed ``logit_bias`` (legacy value 2.65), and
    installs the ``padding_free_fn`` the packed path calls to score the last non-pad token. This is a
    genuine model-method tweak (binding methods onto the instance), hence a ``process_model`` job.
    """

    model_type = 'jina_reranker_m0'
    model_cls = 'transformers:AutoModel'
    trust_remote_code = True
    architectures = ['JinaRerankerM0ForConditionalGeneration']
    template = 'jina_reranker_m0'
    task_type = 'reranker'
    tags = ['vision']
    is_multimodal = True
    models = [('JinaAI/jina-reranker-m0', 'jinaai/jina-reranker-m0')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner=['model.visual.merger'],
            vision_tower=['model.visual'],
        )

    def process_model(self, model):
        from types import MethodType

        from transformers.modeling_outputs import SequenceClassifierOutputWithPast
        if hasattr(model, '_forward_origin'):
            return model
        model._forward_origin = model.forward
        model.logit_bias = 2.65

        def forward(self,
                    input_ids=None,
                    attention_mask=None,
                    position_ids=None,
                    inputs_embeds=None,
                    pixel_values=None,
                    image_grid_thw=None,
                    video_grid_thw=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                    **kwargs):
            kwargs.pop('labels', None)  # ranking models have no LM labels
            if return_dict is None:
                return_dict = True
            out = self._forward_origin(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs)
            logits = out.unsqueeze(-1) - self.logit_bias
            if not return_dict:
                return (logits, )
            return SequenceClassifierOutputWithPast(logits=logits)

        model.forward = MethodType(forward, model)

        def padding_free_fn(self, output, kwargs, padding_side):
            return_dict = kwargs.get('return_dict', None)
            output.logits = output['last_hidden_state'][:, -1]
            logits = self.score(output.logits) - self.logit_bias
            if not return_dict:
                return (logits, )
            return SequenceClassifierOutputWithPast(logits=logits)

        model.padding_free_fn = MethodType(padding_free_fn, model)
        return model
