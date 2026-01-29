# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import Any, Dict

from transformers import AutoModel, PretrainedConfig, PreTrainedModel

from swift.template import TemplateType
from swift.utils import Processor, git_clone_github
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import ModelLoader, register_model
from ..utils import use_submodel_func


class DeepseekLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        # fix dtype bug
        mlp_cls = model.model.layers[1].mlp.__class__

        for module in model.modules():
            if isinstance(module, mlp_cls):
                patch_output_to_input_device(module)
        return model


register_model(
    ModelMeta(
        LLMModelType.deepseek,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-moe-16b-chat', 'deepseek-ai/deepseek-moe-16b-chat'),
                Model('deepseek-ai/deepseek-moe-16b-base', 'deepseek-ai/deepseek-moe-16b-base'),
            ], ),
        ],
        DeepseekLoader,
        template=TemplateType.deepseek,
        architectures=['DeepseekForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_v2,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-Coder-V2-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Instruct'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Base', 'deepseek-ai/DeepSeek-Coder-V2-Base'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Base'),
                Model('deepseek-ai/DeepSeek-V2-Lite', 'deepseek-ai/DeepSeek-V2-Lite'),
                Model('deepseek-ai/DeepSeek-V2-Lite-Chat', 'deepseek-ai/DeepSeek-V2-Lite-Chat'),
                Model('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V2'),
                Model('deepseek-ai/DeepSeek-V2-Chat', 'deepseek-ai/DeepSeek-V2-Chat'),
            ], TemplateType.deepseek),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V2.5', 'deepseek-ai/DeepSeek-V2.5'),
                Model('deepseek-ai/DeepSeek-V2.5-1210', 'deepseek-ai/DeepSeek-V2.5-1210')
            ], TemplateType.deepseek_v2_5)
        ],
        DeepseekLoader,
        model_arch=ModelArch.deepseek_v2,
        architectures=['DeepseekV2ForCausalLM'],
        requires=['transformers>=4.39.3'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_v3,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V3-Base', 'deepseek-ai/DeepSeek-V3-Base'),
                Model('deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-V3'),
                Model('deepseek-ai/DeepSeek-V3-0324', 'deepseek-ai/DeepSeek-V3-0324'),
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('cognitivecomputations/DeepSeek-V3-awq', 'cognitivecomputations/DeepSeek-V3-AWQ'),
                Model('cognitivecomputations/DeepSeek-V3-0324-AWQ', 'cognitivecomputations/DeepSeek-V3-0324-AWQ')
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-Prover-V2-7B', 'deepseek-ai/DeepSeek-Prover-V2-7B'),
                Model('deepseek-ai/DeepSeek-Prover-V2-671B', 'deepseek-ai/DeepSeek-Prover-V2-671B'),
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('unsloth/DeepSeek-V3-bf16', 'unsloth/DeepSeek-V3-bf16'),
                Model('unsloth/DeepSeek-V3-0324-BF16', 'unsloth/DeepSeek-V3-0324-BF16'),
                Model('unsloth/DeepSeek-Prover-V2-671B-BF16', 'unsloth/DeepSeek-Prover-V2-671B-BF16'),
            ], TemplateType.deepseek_v2_5),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-R1'),
                Model('deepseek-ai/DeepSeek-R1-Zero', 'deepseek-ai/DeepSeek-R1-Zero'),
                Model('deepseek-ai/DeepSeek-R1-0528', 'deepseek-ai/DeepSeek-R1-0528'),
            ], TemplateType.deepseek_r1),
            ModelGroup([
                Model('cognitivecomputations/DeepSeek-R1-awq', 'cognitivecomputations/DeepSeek-R1-AWQ'),
                Model('cognitivecomputations/DeepSeek-R1-0528-AWQ', 'cognitivecomputations/DeepSeek-R1-0528-AWQ'),
            ], TemplateType.deepseek_r1),
            ModelGroup([
                Model('unsloth/DeepSeek-R1-BF16', 'unsloth/DeepSeek-R1-BF16'),
                Model('unsloth/DeepSeek-R1-Zero-BF16', 'unsloth/DeepSeek-R1-Zero-BF16'),
                Model('unsloth/DeepSeek-R1-0528-BF16', 'unsloth/DeepSeek-R1-0528-BF16'),
            ], TemplateType.deepseek_r1),
            ModelGroup([
                Model('moonshotai/Moonlight-16B-A3B', 'moonshotai/Moonlight-16B-A3B'),
                Model('moonshotai/Moonlight-16B-A3B-Instruct', 'moonshotai/Moonlight-16B-A3B-Instruct'),
            ], TemplateType.moonlight),
            ModelGroup([
                Model('moonshotai/Kimi-K2-Base', 'moonshotai/Kimi-K2-Base'),
                Model('moonshotai/Kimi-K2-Instruct', 'moonshotai/Kimi-K2-Instruct'),
                Model('moonshotai/Kimi-K2-Instruct-0905', 'moonshotai/Kimi-K2-Instruct-0905'),
                Model('moonshotai/Kimi-K2-Thinking', 'moonshotai/Kimi-K2-Thinking'),
            ], TemplateType.kimi_k2),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V3.1-Base', 'deepseek-ai/DeepSeek-V3.1-Base'),
                Model('deepseek-ai/DeepSeek-V3.1', 'deepseek-ai/DeepSeek-V3.1'),
                Model('deepseek-ai/DeepSeek-V3.1-Terminus', 'deepseek-ai/DeepSeek-V3.1-Terminus'),
            ], TemplateType.deepseek_v3_1),
        ],
        DeepseekLoader,
        model_arch=ModelArch.deepseek_v2,
        architectures=['DeepseekV3ForCausalLM'],
        requires=['transformers>=4.39.3'],
    ))


class DeepseekV32Loader(ModelLoader):

    def get_config(self, model_dir: str):
        try:
            from transformers.models.deepseek_v32 import DeepseekV32Config
        except ImportError:
            from transformers.models.deepseek_v3 import DeepseekV3Config as DeepseekV32Config
        return DeepseekV32Config.from_pretrained(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        try:
            from transformers.models.deepseek_v32 import DeepseekV32ForCausalLM
        except ImportError:
            # It’s only for compatibility with Megatron training or vllm/sglang infer,
            # while we wait for Transformers to support deepseek_v3_2.
            from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM as DeepseekV32ForCausalLM
            if not self.return_dummy_model:
                raise ValueError('DeepSeek-V3.2 is not supported in transformers.')
        self.auto_model_cls = DeepseekV32ForCausalLM
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.deepseek_v3_2,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V3.2', 'deepseek-ai/DeepSeek-V3.2'),
                Model('deepseek-ai/DeepSeek-V3.2-Speciale', 'deepseek-ai/DeepSeek-V3.2-Speciale'),
                Model('deepseek-ai/DeepSeek-V3.2-Exp', 'deepseek-ai/DeepSeek-V3.2-Exp'),
                Model('deepseek-ai/DeepSeek-V3.2-Exp-Base', 'deepseek-ai/DeepSeek-V3.2-Exp-Base'),
                Model('deepseek-ai/DeepSeek-Math-V2', 'deepseek-ai/DeepSeek-Math-V2'),
            ]),
        ],
        DeepseekV32Loader,
        template=TemplateType.deepseek_v3_1,
        architectures=['DeepseekV32ForCausalLM'],
    ))


class DeepseekVLLoader(ModelLoader):

    def get_config(self, model_dir: str):
        # compat with python==3.10
        if sys.version_info.minor >= 10:
            import collections
            import collections.abc
            for type_name in collections.abc.__all__:
                setattr(collections, type_name, getattr(collections.abc, type_name))
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL')
        sys.path.append(local_repo_path)
        from deepseek_vl.models import VLChatProcessor
        self.auto_tokenizer_cls = VLChatProcessor
        return super().get_config(model_dir)

    def _get_model(self, model_dir: str, llm_prefix, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        llm = getattr(model, llm_prefix)
        patch_output_clone(llm.model.embed_tokens)
        patch_output_to_input_device(llm.model.embed_tokens)
        use_submodel_func(model, llm_prefix)
        model.generation_config = llm.generation_config
        return model

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return self._get_model(model_dir, 'language_model', *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-vl-1.3b-chat', 'deepseek-ai/deepseek-vl-1.3b-chat'),
                Model('deepseek-ai/deepseek-vl-7b-chat', 'deepseek-ai/deepseek-vl-7b-chat'),
            ], ),
        ],
        DeepseekVLLoader,
        template=TemplateType.deepseek_vl,
        architectures=['MultiModalityCausalLM'],
        model_arch=ModelArch.deepseek_vl,
        tags=['vision'],
    ))


class DeepseekJanusLoader(DeepseekVLLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return self._get_model(model_dir, 'language_model', *args, **kwargs)

    def get_config(self, model_dir: str):
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/deepseek-ai/Janus')
        sys.path.append(local_repo_path)
        from janus.models import VLChatProcessor
        self.auto_tokenizer_cls = VLChatProcessor
        return super(DeepseekVLLoader, self).get_config(model_dir)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_janus,
        [
            ModelGroup([
                Model('deepseek-ai/Janus-1.3B', 'deepseek-ai/Janus-1.3B'),
            ]),
        ],
        DeepseekJanusLoader,
        template=TemplateType.deepseek_janus,
        model_arch=ModelArch.deepseek_janus,
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.deepseek_janus_pro,
        [
            ModelGroup([
                Model('deepseek-ai/Janus-Pro-1B', 'deepseek-ai/Janus-Pro-1B'),
                Model('deepseek-ai/Janus-Pro-7B', 'deepseek-ai/Janus-Pro-7B'),
            ]),
        ],
        DeepseekJanusLoader,
        template=TemplateType.deepseek_janus_pro,
        model_arch=ModelArch.deepseek_janus,
        tags=['vision'],
    ))


class DeepseekVL2Loader(DeepseekVLLoader):

    def get_config(self, model_dir: str):
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL2')
        sys.path.append(local_repo_path)
        try:
            from deepseek_vl2.models import DeepseekVLV2Processor
        except ImportError:
            # compat transformers>=4.42
            import transformers
            transformers.models.llama.modeling_llama.LlamaFlashAttention2 = None
            from deepseek_vl2.models import DeepseekVLV2Processor
        self.auto_tokenizer_cls = DeepseekVLV2Processor
        return super(DeepseekVLLoader, self).get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return super()._get_model(model_dir, 'language', *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl2,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-vl2-tiny', 'deepseek-ai/deepseek-vl2-tiny'),
                Model('deepseek-ai/deepseek-vl2-small', 'deepseek-ai/deepseek-vl2-small'),
                Model('deepseek-ai/deepseek-vl2', 'deepseek-ai/deepseek-vl2'),
            ]),
        ],
        DeepseekVL2Loader,
        template=TemplateType.deepseek_vl2,
        model_arch=ModelArch.deepseek_vl2,
        requires=['transformers<4.42'],
        tags=['vision'],
    ))


class DeepseekOCRLoader(ModelLoader):
    visual_name = 'vision_model'

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self.auto_model_cls = self.auto_model_cls or AutoModel
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.sam_model)
        patch_output_to_input_device(getattr(model.model, self.visual_name))
        patch_output_to_input_device(model.model.projector)
        return model

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        from transformers import AutoProcessor, AutoTokenizer
        # When not loading model (e.g., vllm backend), avoid triggering AutoConfig which would execute
        # trust_remote_code and cause transformers version compatibility issues
        # For vllm backend, we only need the processor/tokenizer
        try:
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:
            # Fallback to AutoTokenizer if AutoProcessor is not available
            processor = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        return processor


class DeepseekOCR2Loader(DeepseekOCRLoader):
    visual_name = 'qwen2_model'


register_model(
    ModelMeta(
        MLLMModelType.deepseek_ocr,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-OCR', 'deepseek-ai/DeepSeek-OCR'),
            ]),
        ],
        DeepseekOCRLoader,
        template=TemplateType.deepseek_ocr,
        model_arch=ModelArch.deepseek_ocr,
        architectures=['DeepseekOCRForCausalLM'],
        requires=['transformers==4.46.3', 'easydict'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.deepseek_ocr2,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-OCR-2', 'deepseek-ai/DeepSeek-OCR-2'),
            ]),
        ],
        DeepseekOCR2Loader,
        template=TemplateType.deepseek_ocr2,
        model_arch=ModelArch.deepseek_ocr2,
        architectures=['DeepseekOCR2ForCausalLM'],
        requires=['transformers==4.46.3', 'easydict'],
        tags=['vision'],
    ))
