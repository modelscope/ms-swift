# Copyright (c) ModelScope Contributors. All rights reserved.
# Qwen families beyond the plain transformers path; PORT comments mark seams that did not fit.
from __future__ import annotations
import importlib.metadata
import os
from typing import Dict

from packaging import version
from transformers.utils.versions import require_version

from swift.dev.utils import get_logger
from transformers import PretrainedConfig, PreTrainedModel
from ..keep_alive import apply_vision_keep_alive
from .base import ModelArch, ModelLoader, register_model

logger = get_logger()


@register_model
class Qwen2VLLoader(ModelLoader):
    """Qwen2-VL: the root of the whole Qwen vision chain."""

    model_type = 'qwen2_vl'
    model_cls = 'transformers:Qwen2VLForConditionalGeneration'
    architectures = ['Qwen2VLForConditionalGeneration']
    template = 'qwen2_vl'
    requires = ['transformers>=4.45', 'qwen_vl_utils>=0.0.6', 'decord']
    tags = ['vision', 'video']
    is_multimodal = True
    models = [
        'Qwen/Qwen2-VL-2B-Instruct',
        'Qwen/Qwen2-VL-7B-Instruct',
        'Qwen/Qwen2-VL-72B-Instruct',
        'Qwen/Qwen2-VL-2B',
        'Qwen/Qwen2-VL-7B',
        'Qwen/Qwen2-VL-72B',
        'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4',
        'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4',
        'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4',
        'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8',
        'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8',
        'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8',
        'Qwen/Qwen2-VL-2B-Instruct-AWQ',
        'Qwen/Qwen2-VL-7B-Instruct-AWQ',
        'Qwen/Qwen2-VL-72B-Instruct-AWQ',
        'bytedance-research/UI-TARS-2B-SFT',
        'bytedance-research/UI-TARS-7B-SFT',
        'bytedance-research/UI-TARS-7B-DPO',
        'bytedance-research/UI-TARS-72B-SFT',
        'bytedance-research/UI-TARS-72B-DPO',
        'allenai/olmOCR-7B-0225-preview',
        ('OpenDataLab/MinerU2.5-Pro-2604-1.2B', 'opendatalab/MinerU2.5-Pro-2604-1.2B'),
    ]

    # PORT-3: legacy read this defensively off the instance (`getattr(self, 'is_moe', False)`).
    is_moe = False
    # PORT-4: the patch_size a *third-party library* must be configured with -- no seam for it.
    image_patch_size = 14
    # PORT-5: env defaults that only exist to steer qwen_vl_utils. Qwen3VLEmb/Reranker set three.
    env_defaults: Dict[str, str] = {}

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            vision_tower='model.visual',
            aligner='model.visual.merger',
        )

    def setup_environment(self) -> None:
        """PORT-6: a fourth seam the design is missing.

        Legacy called this ``_check_qwen_vl_utils``; it must run *before* the processor is built,
        does version gating, sets env defaults and reaches into ``qwen_vl_utils.vision_process``
        module globals -- neither a ``build`` nor a ``process`` of config/tokenizer/model.
        """
        for key, value in self.env_defaults.items():
            os.environ.setdefault(key, value)
        try:
            utils_version = importlib.metadata.version('qwen_vl_utils')
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError(
                "The 'qwen_vl_utils' distribution was not found and is required by this application.")
        if version.parse(utils_version) >= version.parse('0.0.14'):
            self._compat_qwen_vl_utils(self.image_patch_size)
        else:
            require_version('qwen_vl_utils<0.0.12')

    @staticmethod
    def _compat_qwen_vl_utils(image_patch_size: int) -> None:
        """Translate pixel-count env vars into the token-count ones qwen_vl_utils>=0.0.14 reads.

        The library switched its knobs from pixels to visual tokens; users (and every example
        script) still export ``MAX_PIXELS``. One token covers ``(patch_size * merge)**2`` pixels,
        so the conversion is a plain divide; an already-set token var always wins.
        """
        spatial_merge_size = int(os.getenv('SPATIAL_MERGE_SIZE', '2'))
        image_factor = image_patch_size * spatial_merge_size
        pixels_to_tokens = {
            'MAX_PIXELS': 'IMAGE_MAX_TOKEN_NUM',
            'MIN_PIXELS': 'IMAGE_MIN_TOKEN_NUM',
            'VIDEO_MAX_PIXELS': 'VIDEO_MAX_TOKEN_NUM',
            'VIDEO_MIN_PIXELS': 'VIDEO_MIN_TOKEN_NUM',
        }
        for source_var, target_var in pixels_to_tokens.items():
            value = os.getenv(source_var)
            if value and not os.getenv(target_var):
                os.environ[target_var] = str(int(value) // image_factor**2)

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        # PORT-6 fallout: no dedicated seam, so the environment setup rides the front of
        # build_processor -- a subclass overriding build_processor without super() loses it.
        self.setup_environment()
        return super().build_processor(model_dir, config, **kwargs)

    def process_tokenizer(self, processor):
        from qwen_vl_utils import vision_process

        from ._qwen_vl_utils import patch_qwen_vl_utils
        # In order to have different hashes for the template.
        processor.global_vars = patch_qwen_vl_utils(vision_process)
        return processor

    def process_model(self, model):
        # DeepSpeed ZeRO-3 keep-alive, driven by `ModelArch.aligner`: a text-only micro-batch
        # skips the vision tower and deadlocks the all-gather; see keep_alive.py for the design.
        model._vision_keep_alive = apply_vision_keep_alive(model, self.model_arch.aligner)
        return model


@register_model
class QVQLoader(Qwen2VLLoader):
    """QVQ-72B: Qwen2-VL loading, a reasoning-specific chat template."""

    model_type = 'qvq'
    template = 'qvq'
    architectures = []  # reachable by id / explicit --model_type only; base owns the reverse-lookup
    models = ['Qwen/QVQ-72B-Preview']


@register_model
class Qwen2GmeLoader(Qwen2VLLoader):
    """gme embedding checkpoints: Qwen2-VL loading, the gme template."""

    model_type = 'qwen2_gme'
    template = 'qwen2_gme'
    architectures = []
    tags = ['vision']
    models = [
        ('iic/gme-Qwen2-VL-2B-Instruct', 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct'),
        ('iic/gme-Qwen2-VL-7B-Instruct', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct'),
    ]


@register_model
class Qwen2_5VLLoader(Qwen2VLLoader):
    """Only the model class differs from Qwen2-VL -- one declaration, no method left."""

    model_type = 'qwen2_5_vl'
    model_cls = 'transformers:Qwen2_5_VLForConditionalGeneration'
    architectures = ['Qwen2_5_VLForConditionalGeneration']
    template = 'qwen2_5_vl'
    requires = ['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord']
    models = [
        'Qwen/Qwen2.5-VL-3B-Instruct',
        'Qwen/Qwen2.5-VL-7B-Instruct',
        'Qwen/Qwen2.5-VL-32B-Instruct',
        'Qwen/Qwen2.5-VL-72B-Instruct',
        'Qwen/Qwen2.5-VL-3B-Instruct-AWQ',
        'Qwen/Qwen2.5-VL-7B-Instruct-AWQ',
        'Qwen/Qwen2.5-VL-32B-Instruct-AWQ',
        'Qwen/Qwen2.5-VL-72B-Instruct-AWQ',
    ]


@register_model
class MiMoVLLoader(Qwen2_5VLLoader):
    """Xiaomi MiMo-VL: Qwen2.5-VL loading, its own chat template."""

    model_type = 'mimo_vl'
    template = 'mimo_vl'
    architectures = []
    models = ['XiaomiMiMo/MiMo-VL-7B-SFT', 'XiaomiMiMo/MiMo-VL-7B-RL']


@register_model
class Qwen3VLLoader(Qwen2VLLoader):

    model_type = 'qwen3_vl'
    model_cls = 'transformers:Qwen3VLForConditionalGeneration'
    architectures = ['Qwen3VLForConditionalGeneration']
    template = 'qwen3_vl'
    requires = ['transformers>=4.57', 'qwen_vl_utils>=0.0.14', 'decord']
    image_patch_size = 16
    # PORT-7: requires cannot express "and therefore skip the version fork", so it stays imperative.
    min_qwen_vl_utils = '0.0.14'
    models = [
        'Qwen/Qwen3-VL-2B-Instruct',
        'Qwen/Qwen3-VL-2B-Thinking',
        'Qwen/Qwen3-VL-2B-Instruct-FP8',
        'Qwen/Qwen3-VL-2B-Thinking-FP8',
        'Qwen/Qwen3-VL-4B-Instruct',
        'Qwen/Qwen3-VL-4B-Thinking',
        'Qwen/Qwen3-VL-4B-Instruct-FP8',
        'Qwen/Qwen3-VL-4B-Thinking-FP8',
        'Qwen/Qwen3-VL-8B-Instruct',
        'Qwen/Qwen3-VL-8B-Thinking',
        'Qwen/Qwen3-VL-8B-Instruct-FP8',
        'Qwen/Qwen3-VL-8B-Thinking-FP8',
        'Qwen/Qwen3-VL-32B-Instruct',
        'Qwen/Qwen3-VL-32B-Thinking',
        'Qwen/Qwen3-VL-32B-Instruct-FP8',
        'Qwen/Qwen3-VL-32B-Thinking-FP8',
    ]

    def setup_environment(self) -> None:
        for key, value in self.env_defaults.items():
            os.environ.setdefault(key, value)
        require_version(f'qwen_vl_utils>={self.min_qwen_vl_utils}')
        self._compat_qwen_vl_utils(self.image_patch_size)

    @property
    def model_arch(self) -> ModelArch:
        # Qwen3-VL adds the deepstack mergers, so the aligner is no longer a single prefix; the
        # base `process_model` reads this and hooks all of them -- no override here.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            vision_tower='model.visual',
            aligner=['model.visual.merger', 'model.visual.deepstack_merger_list'],
        )


@register_model
class Qwen3VLMoeLoader(Qwen3VLLoader):

    model_type = 'qwen3_vl_moe'
    model_cls = 'transformers:Qwen3VLMoeForConditionalGeneration'
    architectures = ['Qwen3VLMoeForConditionalGeneration']
    is_moe = True
    models = [
        'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'Qwen/Qwen3-VL-30B-A3B-Thinking',
        'Qwen/Qwen3-VL-30B-A3B-Instruct-FP8',
        'Qwen/Qwen3-VL-30B-A3B-Thinking-FP8',
        'Qwen/Qwen3-VL-235B-A22B-Instruct',
        'Qwen/Qwen3-VL-235B-A22B-Thinking',
        'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8',
        'Qwen/Qwen3-VL-235B-A22B-Thinking-FP8',
    ]


@register_model
class Qwen3_5Loader(Qwen3VLLoader):

    model_type = 'qwen3_5'
    model_cls = 'transformers:Qwen3_5ForConditionalGeneration'
    architectures = ['Qwen3_5ForConditionalGeneration']
    template = 'qwen3_5'
    requires = ['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord']
    models = [
        'Qwen/Qwen3.5-0.8B',
        'Qwen/Qwen3.5-2B',
        'Qwen/Qwen3.5-4B',
        'Qwen/Qwen3.5-9B',
        'Qwen/Qwen3.5-27B',
        'Qwen/Qwen3.5-27B-FP8',
        'Qwen/Qwen3.5-0.8B-Base',
        'Qwen/Qwen3.5-2B-Base',
        'Qwen/Qwen3.5-4B-Base',
        'Qwen/Qwen3.5-9B-Base',
        'Qwen/Qwen3.6-27B',
        'Qwen/Qwen3.6-27B-FP8',
    ]

    @property
    def model_arch(self) -> ModelArch:
        # PORT-9: qwen3_5 uses the qwen2_vl partition (single merger, no deepstack) even though
        # the loader inherits from Qwen3VLLoader -- this is also what keeps the keep-alive correct.
        return Qwen2VLLoader.model_arch.fget(self)


@register_model
class OvisOcr2Loader(Qwen3_5Loader):
    """OvisOCR2: Qwen3.5 loading, an OCR-specific chat template."""

    model_type = 'ovis_ocr2'
    template = 'ovis_ocr2'
    architectures = []
    tags = ['vision']
    models = ['ATH-MaaS/OvisOCR2']


@register_model
class Qwen3_5MoeLoader(Qwen3_5Loader):

    model_type = 'qwen3_5_moe'
    model_cls = 'transformers:Qwen3_5MoeForConditionalGeneration'
    architectures = ['Qwen3_5MoeForConditionalGeneration']
    requires = ['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord']
    is_moe = True
    models = [
        'Qwen/Qwen3.5-35B-A3B-Base',
        'Qwen/Qwen3.5-35B-A3B',
        'Qwen/Qwen3.5-122B-A10B',
        'Qwen/Qwen3.5-397B-A17B',
        'Qwen/Qwen3.5-35B-A3B-FP8',
        'Qwen/Qwen3.5-122B-A10B-FP8',
        'Qwen/Qwen3.5-397B-A17B-FP8',
        'Qwen/Qwen3.6-35B-A3B',
        'Qwen/Qwen3.6-35B-A3B-FP8',
    ]


@register_model
class Qwen3VLEmbLoader(Qwen3VLLoader):
    """Embedding variant: the entire legacy subclass was three env defaults."""

    model_type = 'qwen3_vl_emb'
    template = 'qwen3_vl_emb'
    # Embedding is a different *task* (falling through to `qwen3_vl` generation would be wrong), so
    # the architecture stays declared; reverse-lookup returns qwen3_vl/emb/reranker, caller picks.
    architectures = ['Qwen3VLForConditionalGeneration']
    mcore_model_type = 'qwen3_vl'
    env_defaults = {'IMAGE_MAX_TOKEN_NUM': '1800', 'FPS': '1', 'FPS_MAX_FRAMES': '64'}
    models = ['Qwen/Qwen3-VL-Embedding-2B', 'Qwen/Qwen3-VL-Embedding-8B']


@register_model
class Qwen3VLRerankerLoader(Qwen3VLLoader):

    model_type = 'qwen3_vl_reranker'
    template = 'qwen3_vl_reranker'
    architectures = ['Qwen3VLForConditionalGeneration']  # see Qwen3VLEmbLoader: distinct task, keep it
    mcore_model_type = 'qwen3_vl'
    env_defaults = {'IMAGE_MAX_TOKEN_NUM': '1280', 'FPS': '1', 'FPS_MAX_FRAMES': '64'}
    models = ['Qwen/Qwen3-VL-Reranker-2B', 'Qwen/Qwen3-VL-Reranker-8B']


@register_model
class Qwen2AudioLoader(ModelLoader):
    """Qwen2-Audio: a plain ``Qwen2AudioForConditionalGeneration`` + an audio ``model_arch``. Legacy
    pins ``transformers<4.49`` (an upstream regression window); ported faithfully, so the family is
    flagged on dev's transformers 5.5 even though the class is present in-tree."""

    model_type = 'qwen2_audio'
    model_cls = 'transformers:Qwen2AudioForConditionalGeneration'
    architectures = ['Qwen2AudioForConditionalGeneration']
    template = 'qwen2_audio'
    requires = ['transformers>=4.45,<4.49', 'librosa']
    tags = ['audio']
    is_multimodal = True
    models = [
        'Qwen/Qwen2-Audio-7B-Instruct',
        'Qwen/Qwen2-Audio-7B',
    ]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.qwen2_audio` (not version-gated); bare top-level parts.
        return ModelArch(
            language_model='language_model',
            aligner='multi_modal_projector',
            vision_tower='audio_tower',
        )


# AIDC-AI Ovis family (remote-code ``Ovis`` / ``Ovis2_5``); unrelated to OvisOcr2Loader above,
# which is a Qwen3.5 template variant sharing only the "ovis" brand.


@register_model
class OvisLoader(ModelLoader):
    """Ovis1.6 / Ovis2: a thin wrapper whose real LLM is ``model.llm``, so top-level calls are
    delegated there. Legacy also cloned the embedding output and forced the vit's input embeddings
    -- both obsolete per PATCH_INVENTORY, dropped. The remaining seams are real: the visual
    tokenizer / VTE must match the model dtype, and Ovis's static KV cache is disabled. Legacy's
    ``attn_impl_keys=['llm_attn_implementation']`` routing is a known difference not wired here."""

    model_type = 'ovis1_6'
    model_cls = 'transformers:AutoModelForCausalLM'
    processor_cls = 'transformers:AutoTokenizer'  # legacy forced AutoTokenizer, not AutoProcessor
    trust_remote_code = True
    architectures = ['Ovis']
    template = 'ovis1_6'
    requires = ['transformers>=4.42']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('AIDC-AI/Ovis1.6-Gemma2-9B', 'AIDC-AI/Ovis1.6-Gemma2-9B'),
        ('AIDC-AI/Ovis1.6-Gemma2-9B-GPTQ-Int4', 'AIDC-AI/Ovis1.6-Gemma2-9B-GPTQ-Int4'),
        ('AIDC-AI/Ovis1.6-Gemma2-27B', 'AIDC-AI/Ovis1.6-Gemma2-27B'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model='llm', vision_tower=['visual_tokenizer', 'vte'])

    def process_model(self, model):
        model.visual_tokenizer.to(model.dtype)
        model.vte.to(model.dtype)
        model.generation_config.cache_implementation = None
        self.delegate_to_submodel(model, 'llm', ['generate', 'forward', 'get_input_embeddings'])
        return model


@register_model
class Ovis1_6Llama3Loader(OvisLoader):
    model_type = 'ovis1_6_llama3'
    template = 'ovis1_6_llama3'
    architectures = []  # template variant of ovis1_6 (same ``Ovis`` class)
    models = [('AIDC-AI/Ovis1.6-Llama3.2-3B', 'AIDC-AI/Ovis1.6-Llama3.2-3B')]


@register_model
class Ovis2Loader(OvisLoader):
    # A distinct model_type, not a template variant: it keeps ``architectures=['Ovis']`` so reverse
    # lookup returns ovis1_6 / ovis2 together (both use the ``Ovis`` class; id disambiguates).
    model_type = 'ovis2'
    template = 'ovis2'
    requires = ['transformers>=4.46.2', 'moviepy<2']
    models = [
        ('AIDC-AI/Ovis2-1B', 'AIDC-AI/Ovis2-1B'),
        ('AIDC-AI/Ovis2-2B', 'AIDC-AI/Ovis2-2B'),
        ('AIDC-AI/Ovis2-4B', 'AIDC-AI/Ovis2-4B'),
        ('AIDC-AI/Ovis2-8B', 'AIDC-AI/Ovis2-8B'),
        ('AIDC-AI/Ovis2-16B', 'AIDC-AI/Ovis2-16B'),
        ('AIDC-AI/Ovis2-34B', 'AIDC-AI/Ovis2-34B'),
    ]


@register_model
class Ovis2_5Loader(ModelLoader):
    """Ovis2.5: same ``model.llm`` delegation and dtype alignment, different ``model_arch`` and no
    cache-implementation override."""

    model_type = 'ovis2_5'
    model_cls = 'transformers:AutoModelForCausalLM'
    processor_cls = 'transformers:AutoTokenizer'
    trust_remote_code = True
    architectures = ['Ovis2_5']
    template = 'ovis2_5'
    requires = ['transformers>=4.46.2', 'moviepy<2']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('AIDC-AI/Ovis2.5-2B', 'AIDC-AI/Ovis2.5-2B'),
        ('AIDC-AI/Ovis2.5-9B', 'AIDC-AI/Ovis2.5-9B'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model='llm',
            aligner='visual_tokenizer.head',
            vision_tower=['visual_tokenizer.vit', 'vte'],
        )

    def process_model(self, model):
        model.visual_tokenizer.to(model.dtype)
        model.vte.to(model.dtype)
        self.delegate_to_submodel(model, 'llm', ['generate', 'forward', 'get_input_embeddings'])
        return model


# Qwen text families: one-template-per-loader like llama.py (reverse-lookup owner + architectures=[]
# template variants). thinking/nothinking/coder recur across dense/moe/next; dense owns bare names.


@register_model
class Qwen2Loader(ModelLoader):
    """Qwen1.5 / Qwen2 / Qwen2.5-1M; reverse-lookup owner for ``Qwen2ForCausalLM``."""

    model_type = 'qwen2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Qwen2ForCausalLM']
    template = 'qwen'
    requires = ['transformers>=4.37']
    models = [
        'Qwen/Qwen1.5-0.5B-Chat', 'Qwen/Qwen1.5-1.8B-Chat', 'Qwen/Qwen1.5-4B-Chat', 'Qwen/Qwen1.5-7B-Chat',
        'Qwen/Qwen1.5-14B-Chat', 'Qwen/Qwen1.5-32B-Chat', 'Qwen/Qwen1.5-72B-Chat', 'Qwen/Qwen1.5-110B-Chat',
        'Qwen/Qwen1.5-0.5B', 'Qwen/Qwen1.5-1.8B', 'Qwen/Qwen1.5-4B', 'Qwen/Qwen1.5-7B', 'Qwen/Qwen1.5-14B',
        'Qwen/Qwen1.5-32B', 'Qwen/Qwen1.5-72B', 'Qwen/Qwen1.5-110B',
        'Qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-72B-Instruct',
        'Qwen/Qwen2-0.5B', 'Qwen/Qwen2-1.5B', 'Qwen/Qwen2-7B', 'Qwen/Qwen2-72B',
        'Qwen/Qwen2.5-7B-Instruct-1M', 'Qwen/Qwen2.5-14B-Instruct-1M',
        ('PowerInfer/SmallThinker-3B-Preview', 'PowerInfer/SmallThinker-3B-Preview'),
    ]


@register_model
class CodeQwenLoader(Qwen2Loader):
    model_type = 'code_qwen'
    architectures = []
    template = 'qwen'
    tags = ['coding']
    models = [
        'Qwen/CodeQwen1.5-7B', 'Qwen/CodeQwen1.5-7B-Chat', 'Qwen/CodeQwen1.5-7B-Chat-AWQ',
    ]


@register_model
class Qwen2MathLoader(Qwen2Loader):
    model_type = 'qwen2_math'
    architectures = []
    template = 'qwen'
    tags = ['math']
    models = [
        'Qwen/Qwen2-Math-1.5B-Instruct', 'Qwen/Qwen2-Math-7B-Instruct', 'Qwen/Qwen2-Math-72B-Instruct',
        'Qwen/Qwen2-Math-1.5B', 'Qwen/Qwen2-Math-7B', 'Qwen/Qwen2-Math-72B',
    ]


@register_model
class Qwen2_5Loader(Qwen2Loader):
    model_type = 'qwen2_5'
    architectures = []
    template = 'qwen2_5'
    models = [
        'Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct',
        'Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B',
        'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-72B',
        ('moonshotai/Kimi-Dev-72B', 'moonshotai/Kimi-Dev-72B'),
    ]


@register_model
class Qwen2_5CoderLoader(Qwen2Loader):
    model_type = 'qwen2_5_coder'
    architectures = []
    template = 'qwen2_5'
    tags = ['coding']
    models = [
        'Qwen/Qwen2.5-Coder-0.5B-Instruct', 'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'Qwen/Qwen2.5-Coder-3B-Instruct',
        'Qwen/Qwen2.5-Coder-7B-Instruct', 'Qwen/Qwen2.5-Coder-14B-Instruct', 'Qwen/Qwen2.5-Coder-32B-Instruct',
        'Qwen/Qwen2.5-Coder-0.5B', 'Qwen/Qwen2.5-Coder-1.5B', 'Qwen/Qwen2.5-Coder-3B', 'Qwen/Qwen2.5-Coder-7B',
        'Qwen/Qwen2.5-Coder-14B', 'Qwen/Qwen2.5-Coder-32B',
    ]


@register_model
class Qwen2_5MathLoader(Qwen2Loader):
    model_type = 'qwen2_5_math'
    architectures = []
    template = 'qwen2_5_math'
    tags = ['math']
    models = [
        'Qwen/Qwen2.5-Math-1.5B-Instruct', 'Qwen/Qwen2.5-Math-7B-Instruct', 'Qwen/Qwen2.5-Math-72B-Instruct',
        'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-72B',
    ]


@register_model
class MarcoO1Loader(Qwen2Loader):
    model_type = 'marco_o1'
    architectures = []
    template = 'marco_o1'
    models = [('AIDC-AI/Marco-o1', 'AIDC-AI/Marco-o1')]


@register_model
class QwQPreviewLoader(Qwen2Loader):
    model_type = 'qwq_preview'
    architectures = []
    template = 'qwq_preview'
    models = [('Qwen/QwQ-32B-Preview', 'Qwen/QwQ-32B-Preview')]


@register_model
class QwQLoader(Qwen2Loader):
    model_type = 'qwq'
    architectures = []
    template = 'qwq'
    models = ['Qwen/QwQ-32B', 'Qwen/QwQ-32B-AWQ']


@register_model
class Qwen2MoeLoader(ModelLoader):
    """Qwen1.5-MoE / Qwen2-57B-A14B; reverse-lookup owner for ``Qwen2MoeForCausalLM``."""

    model_type = 'qwen2_moe'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Qwen2MoeForCausalLM']
    template = 'qwen'
    requires = ['transformers>=4.40']
    is_moe = True
    models = [
        'Qwen/Qwen1.5-MoE-A2.7B-Chat', 'Qwen/Qwen1.5-MoE-A2.7B',
        'Qwen/Qwen2-57B-A14B-Instruct', 'Qwen/Qwen2-57B-A14B',
    ]


@register_model
class Qwen3Loader(ModelLoader):
    """Qwen3 dense; reverse-lookup owner for ``Qwen3ForCausalLM``. Owns the bare thinking template
    names (moe/next reuse the templates under family-qualified model_types)."""

    model_type = 'qwen3'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Qwen3ForCausalLM']
    template = 'qwen3'
    requires = ['transformers>=4.51']
    models = [
        'Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-4B-Base', 'Qwen/Qwen3-8B-Base',
        'Qwen/Qwen3-14B-Base', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B',
        'Qwen/Qwen3-14B', 'Qwen/Qwen3-32B',
        # quantized releases (same architecture/template; the quant config rides in the checkpoint)
        'Qwen/Qwen3-0.6B-FP8', 'Qwen/Qwen3-1.7B-FP8', 'Qwen/Qwen3-4B-FP8', 'Qwen/Qwen3-8B-FP8',
        'Qwen/Qwen3-14B-FP8', 'Qwen/Qwen3-32B-FP8',
        'Qwen/Qwen3-4B-AWQ', 'Qwen/Qwen3-8B-AWQ', 'Qwen/Qwen3-14B-AWQ', 'Qwen/Qwen3-32B-AWQ',
        'swift/Qwen3-32B-AWQ',
    ]


@register_model
class Qwen3GuardLoader(Qwen3Loader):
    model_type = 'qwen3_guard'
    architectures = []
    template = 'qwen3_guard'
    models = ['Qwen/Qwen3Guard-Gen-0.6B', 'Qwen/Qwen3Guard-Gen-4B', 'Qwen/Qwen3Guard-Gen-8B']


@register_model
class YuFengXGuardLoader(Qwen3Loader):
    model_type = 'yufeng_xguard'
    architectures = []
    template = 'yufeng_xguard'
    models = [
        'Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B', 'Alibaba-AAIG/YuFeng-XGuard-Reason-8B',
    ]


@register_model
class Qwen3ThinkingLoader(Qwen3Loader):
    model_type = 'qwen3_thinking'
    architectures = []
    template = 'qwen3_thinking'
    models = ['Qwen/Qwen3-4B-Thinking-2507', 'Qwen/Qwen3-4B-Thinking-2507-FP8']


@register_model
class Qwen3NoThinkingLoader(Qwen3Loader):
    model_type = 'qwen3_nothinking'
    architectures = []
    template = 'qwen3_nothinking'
    models = ['Qwen/Qwen3-4B-Instruct-2507', 'Qwen/Qwen3-4B-Instruct-2507-FP8']


@register_model
class Qwen3MoeLoader(ModelLoader):
    """Qwen3-MoE; reverse-lookup owner for ``Qwen3MoeForCausalLM``."""

    model_type = 'qwen3_moe'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Qwen3MoeForCausalLM']
    template = 'qwen3'
    requires = ['transformers>=4.51']
    is_moe = True
    models = [
        'Qwen/Qwen3-30B-A3B-Base', 'Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-235B-A22B',
        ('iic/Tongyi-DeepResearch-30B-A3B', 'Alibaba-NLP/Tongyi-DeepResearch-30B-A3B'),
        # quantized releases
        'Qwen/Qwen3-30B-A3B-FP8', 'Qwen/Qwen3-235B-A22B-FP8',
        ('swift/Qwen3-30B-A3B-AWQ', 'cognitivecomputations/Qwen3-30B-A3B-AWQ'),
        ('swift/Qwen3-235B-A22B-AWQ', 'cognitivecomputations/Qwen3-235B-A22B-AWQ'),
    ]


@register_model
class Qwen3MoeNoThinkingLoader(Qwen3MoeLoader):
    model_type = 'qwen3_moe_nothinking'
    architectures = []
    template = 'qwen3_nothinking'
    models = [
        'Qwen/Qwen3-30B-A3B-Instruct-2507', 'Qwen/Qwen3-235B-A22B-Instruct-2507',
        ('AIDC-AI/Marco-Nano-Base', 'AIDC-AI/Marco-Nano-Base'),
        ('AIDC-AI/Marco-Mini-Instruct', 'AIDC-AI/Marco-Mini-Instruct'),
        ('AIDC-AI/Marco-Nano-Instruct', 'AIDC-AI/Marco-Nano-Instruct'),
        ('AIDC-AI/Marco-Mini-Base', 'AIDC-AI/Marco-Mini-Base'),
        ('AIDC-AI/Marco-Mini-Global-Base', 'AIDC-AI/Marco-Mini-Global-Base'),
        # quantized releases
        'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8', 'Qwen/Qwen3-235B-A22B-Instruct-2507-FP8',
        'swift/Qwen3-235B-A22B-Instruct-2507-AWQ',
    ]


@register_model
class Qwen3MoeThinkingLoader(Qwen3MoeLoader):
    model_type = 'qwen3_moe_thinking'
    architectures = []
    template = 'qwen3_thinking'
    models = [
        'Qwen/Qwen3-30B-A3B-Thinking-2507', 'Qwen/Qwen3-235B-A22B-Thinking-2507',
        ('iic/QwenLong-L1.5-30B-A3B', 'Tongyi-Zhiwen/QwenLong-L1.5-30B-A3B'),
        # quantized releases
        'Qwen/Qwen3-30B-A3B-Thinking-2507-FP8', 'Qwen/Qwen3-235B-A22B-Thinking-2507-FP8',
        'swift/Qwen3-235B-A22B-Thinking-2507-AWQ',
    ]


@register_model
class Qwen3MoeCoderLoader(Qwen3MoeLoader):
    model_type = 'qwen3_moe_coder'
    architectures = []
    template = 'qwen3_coder'
    tags = ['coding']
    models = [
        'Qwen/Qwen3-Coder-30B-A3B-Instruct', 'Qwen/Qwen3-Coder-480B-A35B-Instruct',
        # quantized releases
        'Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8', 'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8',
        'swift/Qwen3-Coder-480B-A35B-Instruct-AWQ',
    ]


@register_model
class Qwen3NextLoader(ModelLoader):
    """Qwen3-Next; reverse-lookup owner for ``Qwen3NextForCausalLM``. Its base group is the
    ``qwen3_nothinking`` (Instruct) template."""

    model_type = 'qwen3_next'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Qwen3NextForCausalLM']
    template = 'qwen3_nothinking'
    requires = ['transformers>=4.57']
    is_moe = True
    models = ['Qwen/Qwen3-Next-80B-A3B-Instruct', 'Qwen/Qwen3-Next-80B-A3B-Instruct-FP8']


@register_model
class Qwen3NextThinkingLoader(Qwen3NextLoader):
    model_type = 'qwen3_next_thinking'
    architectures = []
    template = 'qwen3_thinking'
    models = ['Qwen/Qwen3-Next-80B-A3B-Thinking', 'Qwen/Qwen3-Next-80B-A3B-Thinking-FP8']


@register_model
class Qwen3NextCoderLoader(Qwen3NextLoader):
    model_type = 'qwen3_next_coder'
    architectures = []
    template = 'qwen3_coder'
    tags = ['coding']
    models = [
        'Qwen/Qwen3-Coder-Next-Base', 'Qwen/Qwen3-Coder-Next', 'Qwen/Qwen3-Coder-Next-FP8',
    ]


@register_model
class Qwen3EmbLoader(Qwen3Loader):
    """Qwen3-Embedding: a plain ``Qwen3ForCausalLM`` checkpoint (legacy declared no loader at all),
    distinguished only by its template and its mcore counterpart. Embedding is a different *task*,
    so the architecture stays declared (falling through to ``qwen3`` generation would be wrong);
    reverse-lookup returns qwen3 / qwen3_emb / qwen3_reranker together and the caller picks by id.
    ``task_type`` is deliberately unpinned, exactly as legacy (the head is the user's choice);
    legacy's ``additional_saved_files`` is an export-time concern dev's loader does not model."""

    model_type = 'qwen3_emb'
    architectures = ['Qwen3ForCausalLM']
    template = 'qwen3_emb'
    mcore_model_type = 'qwen3_emb'
    models = ['Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-8B']


@register_model
class Qwen3RerankerLoader(Qwen3Loader):
    """Qwen3-Reranker: like ``Qwen3EmbLoader``, a bare ``Qwen3ForCausalLM`` + its own template. The
    mcore counterpart is plain ``gpt`` (legacy), since reranking rides the generic backbone."""

    model_type = 'qwen3_reranker'
    architectures = ['Qwen3ForCausalLM']  # see Qwen3EmbLoader: distinct task, keep it declared
    template = 'qwen3_reranker'
    mcore_model_type = 'gpt'
    models = ['Qwen/Qwen3-Reranker-0.6B', 'Qwen/Qwen3-Reranker-4B', 'Qwen/Qwen3-Reranker-8B']


@register_model
class MidashengLMLoader(ModelLoader):
    """MiDashengLM: an audio LM (Dasheng encoder + Qwen2.5 decoder), remote-code ``MiDashengLMModel``
    (absent from transformers 5.5) so it rides ``AutoModel`` + ``trust_remote_code``. The Dasheng
    encoder is numerically unstable in bf16/fp16, so ``audio_encoder.float()`` is kept (a real
    requirement, not a placement patch); legacy's ``patch_output_clone`` is dropped."""

    model_type = 'midashenglm'
    model_cls = 'transformers:AutoModel'
    trust_remote_code = True
    architectures = ['MiDashengLMModel']
    template = 'midashenglm'
    requires = ['transformers>=4.52', 'soundfile']
    tags = ['audio']
    is_multimodal = True
    models = [('mispeech/midashenglm-7b', 'mispeech/midashenglm-7b')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model='decoder',
            aligner='audio_projector',
            vision_tower='audio_encoder',
        )

    def process_model(self, model):
        model.audio_encoder.float()
        return model


# Task variants: embedding / reward / PRM.


@register_model
class Qwen2GteLoader(Qwen2Loader):
    """gte-Qwen2: a ``Qwen2ForCausalLM`` checkpoint served as a sentence embedder. Legacy loaded it
    through ``SentenceTransformersLoader``; on dev that pipeline construction lives at the model
    layer (sentence_transformer_model.py), so the loader only carries registration facts, pins the
    embedding task, and marks the framework. Architecture stays declared (a task variant, like
    ``Qwen3EmbLoader``): a gte checkpoint is always embedding, so reverse-lookup returns qwen2 /
    qwen2_gte together and the caller disambiguates by id."""

    model_type = 'qwen2_gte'
    architectures = ['Qwen2ForCausalLM']
    template = 'dummy'
    task_type = 'embedding'
    model_framework = 'sentence_transformers'
    trust_remote_code = True
    models = [
        ('iic/gte_Qwen2-1.5B-instruct', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'),
        ('iic/gte_Qwen2-7B-instruct', 'Alibaba-NLP/gte-Qwen2-7B-instruct'),
    ]


@register_model
class Qwen2RewardLoader(ModelLoader):
    """Qwen2-Math-RM: a reward model. ``Qwen2ForRewardModel`` is not in transformers 5.5 -- the
    checkpoint ships the class via remote code and its ``auto_map`` maps ``AutoModel`` to it, so it
    loads with ``AutoModel`` + ``trust_remote_code``. ``is_reward`` marks the num_labels=1 reward
    head; task resolution still lands on seq_cls. Reverse-lookup owner for ``Qwen2ForRewardModel``."""

    model_type = 'qwen2_reward'
    model_cls = 'transformers:AutoModel'
    architectures = ['Qwen2ForRewardModel']
    template = 'qwen'
    trust_remote_code = True
    is_reward = True
    requires = ['transformers>=4.37']
    models = [('Qwen/Qwen2-Math-RM-72B', 'Qwen/Qwen2-Math-RM-72B')]


@register_model
class Qwen2_5MathRewardLoader(Qwen2RewardLoader):
    """Qwen2.5-Math-RM: same reward head as ``Qwen2RewardLoader``, different (math) template."""

    model_type = 'qwen2_5_math_reward'
    architectures = []  # template variant of the qwen2_reward architecture; owner stays `qwen2_reward`
    template = 'qwen2_5_math'
    models = [('Qwen/Qwen2.5-Math-RM-72B', 'Qwen/Qwen2.5-Math-RM-72B')]


@register_model
class Qwen2_5PRMLoader(ModelLoader):
    """Qwen2.5-Math-PRM: a *process* reward model scoring each reasoning step (a per-token 2-class
    head), distinct from a sequence-level reward model; not in transformers 5.5, so it loads through
    remote code like the RM. dev's builders have no ``prm`` task branch (a PRM's per-step head is
    neither a causal-LM head nor a single seq_cls head), so it loads natively via ``AutoModel``,
    keeps ``is_reward`` for the num_labels handling, but does NOT pin ``task_type='prm``';
    a dedicated PRM step-level loss is a separate builder-side task."""

    model_type = 'qwen2_5_prm'
    model_cls = 'transformers:AutoModel'
    architectures = ['Qwen2ForProcessRewardModel']
    template = 'qwen2_5_math_prm'
    trust_remote_code = True
    is_reward = True
    requires = ['transformers>=4.37']
    models = [
        ('Qwen/Qwen2.5-Math-PRM-7B', 'Qwen/Qwen2.5-Math-PRM-7B'),
        ('Qwen/Qwen2.5-Math-7B-PRM800K', 'Qwen/Qwen2.5-Math-7B-PRM800K'),
        ('Qwen/Qwen2.5-Math-PRM-72B', 'Qwen/Qwen2.5-Math-PRM-72B'),
    ]


# Qwen-Omni (thinker + talker).


class _QwenOmniLoader(Qwen2VLLoader):
    """Shared plumbing for the Qwen-Omni families (thinker LM + talker/token2wav generator).

    Reuses Qwen2-VL's env setup but points ``process_tokenizer`` at ``qwen_omni_utils.vision_process``;
    the thin wrapper's real LM is ``.thinker``, so ``process_model`` proxies there. Dropped, per
    PATCH_INVENTORY: ``_no_split_modules`` / ``patch_get_input_embeddings`` (in-tree Omni declares
    its own leaves) and the AWQ base_model special-case.
    """

    def process_config(self, config):
        # Opt-in audio decoder (talker/token2wav). Off unless ENABLE_AUDIO_OUTPUT is set.
        enable = os.getenv('ENABLE_AUDIO_OUTPUT')
        if enable is not None:
            config.enable_audio_output = enable.lower() in ('1', 'true', 'yes', 'on')
        return config

    def process_tokenizer(self, processor):
        from qwen_omni_utils import vision_process

        from ._qwen_vl_utils import patch_qwen_vl_utils
        processor.global_vars = patch_qwen_vl_utils(vision_process)
        return processor

    def process_model(self, model):
        self.delegate_to_submodel(model, 'thinker')
        # These intermediates are not returnable tensors; keep them out of the inference outputs.
        model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        model.config.talker_config.pad_token_id = None
        return model


@register_model
class Qwen2_5OmniLoader(_QwenOmniLoader):
    model_type = 'qwen2_5_omni'
    model_cls = 'transformers:Qwen2_5OmniForConditionalGeneration'
    processor_cls = 'transformers:Qwen2_5OmniProcessor'
    architectures = ['Qwen2_5OmniModel', 'Qwen2_5OmniForConditionalGeneration']
    template = 'qwen2_5_omni'
    requires = ['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord']
    tags = ['vision', 'video', 'audio']
    is_multimodal = True
    image_patch_size = 14
    # spk_dict.pt (speaker embeddings) is not a standard weight file, so pull it explicitly; the
    # empty ignore list also disables the default download skips.
    additional_saved_files = ['spk_dict.pt']
    ignore_patterns = []
    models = [('Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-3B'), ('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['thinker.model', 'thinker.lm_head'],
            aligner=['thinker.audio_tower.proj', 'thinker.visual.merger'],
            vision_tower=['thinker.audio_tower', 'thinker.visual'],
            generator=['talker', 'token2wav'],
        )


@register_model
class Qwen3OmniMoeLoader(_QwenOmniLoader):
    """Qwen3-Omni-MoE. Legacy forked ``forward`` (``_compat_qwen3_omni_mixed_data``) to inject dummy
    media under DeepSpeed; dev does that without forking forward, via the data-side dummy +
    ``apply_vision_keep_alive`` aligner hooks. The audio-pad token id, which legacy set on the
    config from the tokenizer, is set in ``build_processor`` (the hook holding both objects)."""

    model_type = 'qwen3_omni_moe'
    model_cls = 'transformers:Qwen3OmniMoeForConditionalGeneration'
    processor_cls = 'transformers:Qwen3OmniMoeProcessor'
    architectures = ['Qwen3OmniMoeForConditionalGeneration']
    template = 'qwen3_omni'
    requires = ['transformers>=4.57.dev0', 'soundfile', 'decord', 'qwen_omni_utils>=0.0.9']
    tags = ['vision', 'video', 'audio']
    is_multimodal = True
    is_moe = True
    image_patch_size = 16
    models = [
        'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        'Qwen/Qwen3-Omni-30B-A3B-Thinking',
        'Qwen/Qwen3-Omni-30B-A3B-Captioner',
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['thinker.model', 'thinker.lm_head'],
            aligner=['thinker.audio_tower.proj1', 'thinker.audio_tower.proj2', 'thinker.visual.merger',
                     'thinker.visual.merger_list'],
            vision_tower=['thinker.audio_tower', 'thinker.visual'],
            generator=['talker', 'code2wav'],
        )

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        processor = super().build_processor(model_dir, config, **kwargs)
        config.thinker_config.audio_token_id = processor.tokenizer.encode('<|audio_pad|>')[0]
        return processor

    def process_model(self, model):
        model = super().process_model(model)
        model._vision_keep_alive = apply_vision_keep_alive(model, self.model_arch.aligner)
        return model


# Qwen3 speech (external packages): the classes ship in the `qwen-asr` / `qwen-tts` pip packages;
# legacy's transformers pins guard *the package's* compatibility, not the checkpoint's.


@register_model
class Qwen3ASRLoader(ModelLoader):
    """Qwen3-ASR. ``Qwen3ASRForConditionalGeneration`` comes from the ``qwen-asr`` package, whose
    ``auto_map`` wiring makes ``AutoModel`` resolve it once imported -- hence the import in
    ``build_config``. The ``transformers==4.57.6`` pin is relaxed via ``compat_check_model_inputs``:
    5.x turned ``@check_model_inputs()`` (factory) into a bare decorator, breaking the package
    import; the shim accepts both spellings (verified on 5.5.0, calling convention only; auxiliary
    outputs unverified). ``process_model`` delegates to ``thinker``.
    """

    model_type = 'qwen3_asr'
    model_cls = 'transformers:AutoModel'
    architectures = ['Qwen3ASRForConditionalGeneration']
    template = 'qwen3_asr'
    requires = ['qwen-asr']
    tags = ['audio']
    is_multimodal = True
    models = [('Qwen/Qwen3-ASR-1.7B', 'Qwen/Qwen3-ASR-1.7B'), ('Qwen/Qwen3-ASR-0.6B', 'Qwen/Qwen3-ASR-0.6B')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['thinker.model', 'thinker.lm_head'],
            aligner=['thinker.audio_tower.proj1', 'thinker.audio_tower.proj2'],
            vision_tower=['thinker.audio_tower'],
        )

    @staticmethod
    def compat_check_model_inputs():
        """Let ``check_model_inputs`` accept the transformers 4.x ``@deco()`` spelling.

        4.57 exposed a decorator *factory*; 5.x a plain decorator, so 4.57-era third-party model
        code fails at import. The wrapper dispatches on "was I handed the function or not"
        (idempotent). Patching ``transformers.utils.generic`` suffices on 5.5, and it runs before
        ``qwen_asr`` imports the symbol, so the package binds the wrapper.
        """
        import transformers.utils.generic as generic
        origin = getattr(generic, 'check_model_inputs', None)
        if origin is None or getattr(origin, '_swift_compat', False):
            return

        def check_model_inputs(*args, **kwargs):
            if len(args) == 1 and not kwargs and callable(args[0]):
                return origin(args[0])  # bare @check_model_inputs
            return lambda func: origin(func)  # @check_model_inputs(...)

        check_model_inputs._swift_compat = True
        generic.check_model_inputs = check_model_inputs

    def build_config(self, model_dir: str, **kwargs):
        self.compat_check_model_inputs()
        import qwen_asr  # noqa: F401  # registers Qwen3ASR* into the Auto* mappings
        return super().build_config(model_dir, **kwargs)

    def process_model(self, model):
        self.delegate_to_submodel(model, 'thinker')
        return model


@register_model
class Qwen3TTSLoader(ModelLoader):
    """Qwen3-TTS. Unlike ASR, ``qwen-tts`` does *not* self-register: ``build_config`` wires its
    three classes into the Auto* maps under ``qwen3_tts`` (idempotent per key), as legacy did.
    Training runs through twinkle's ``Qwen3TTSTrainingPatch`` (dual-channel step, speaker embedding
    at codec position 6, 15 sub-talker embeddings + cross-entropy), applied persistently to match
    legacy's forward swap; ``speaker_encoder`` frozen; embeddings/gc delegated to ``talker``. The
    ``<5`` ceiling is dropped; **package not installed here**, expect an ASR-like compat shim.
    """

    model_type = 'qwen3_tts'
    model_cls = 'transformers:AutoModel'
    architectures = ['Qwen3TTSForConditionalGeneration']
    template = 'qwen3_tts'
    requires = ['qwen-tts']
    tags = ['audio', 'tts']
    is_multimodal = True
    models = [
        'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
        'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
        'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model=['talker'], generator=['speaker_encoder'])

    def build_config(self, model_dir: str, **kwargs):
        from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

        from transformers import AutoConfig, AutoModel, AutoProcessor
        AutoConfig.register('qwen3_tts', Qwen3TTSConfig, exist_ok=True)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor, exist_ok=True)
        return super().build_config(model_dir, **kwargs)

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        from qwen_tts import Qwen3TTSTokenizer

        from swift.dev.utils import safe_snapshot_download
        from swift.dev.utils import get_env_args
        processor = super().build_processor(model_dir, config, **kwargs)
        tts_path = get_env_args('tts_tokenizer_path', str, 'Qwen/Qwen3-TTS-Tokenizer-12Hz')
        tokenizer_path = safe_snapshot_download(tts_path)
        processor.tts_tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_path, device_map='cpu')
        return processor

    def process_model(self, model):
        from twinkle.patch import apply_patch

        from swift.dev.patch.transformers_qwen3_tts import Qwen3TTSTrainingPatch
        self.delegate_to_submodel(model, 'talker', ['get_input_embeddings', 'gradient_checkpointing_enable'])
        if model.speaker_encoder is not None:
            for param in model.speaker_encoder.parameters():
                param.requires_grad = False
        # Applied persistently (not via apply_context) to match legacy's permanent forward swap.
        apply_patch(model, Qwen3TTSTrainingPatch())
        return model
