# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import sys
import torch
import types
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from typing import Any, Dict

from swift.template import TemplateType
from swift.utils import Processor, get_logger, git_clone_github
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
        mlp_cls = model.model.layers[-1].mlp.__class__

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
            ],
                       TemplateType.moonlight,
                       requires=['transformers<4.49']),
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
            # while we wait for Transformers to support deepseek_v32.
            from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM as DeepseekV32ForCausalLM
            if not self.return_dummy_model:
                raise ValueError('DeepSeek-V3.2 is not supported in transformers.')
        self.auto_model_cls = DeepseekV32ForCausalLM
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.deepseek_v32,
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

register_model(
    ModelMeta(
        LLMModelType.deepseek_v4,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V4-Flash', 'deepseek-ai/DeepSeek-V4-Flash'),
                Model('deepseek-ai/DeepSeek-V4-Flash-Base', 'deepseek-ai/DeepSeek-V4-Flash-Base'),
            ]),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V4-Pro', 'deepseek-ai/DeepSeek-V4-Pro'),
                Model('deepseek-ai/DeepSeek-V4-Pro-Base', 'deepseek-ai/DeepSeek-V4-Pro-Base'),
            ]),
        ],
        template=TemplateType.deepseek_v4,
        architectures=['DeepseekV4ForCausalLM'],
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


class UnlimitedOCRLoader(DeepseekOCRLoader):
    visual_name = 'vision_model'

    @staticmethod
    def _make_rswa_causal_mask(
        seq_len: int,
        prefill_lens: torch.Tensor,
        window_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Constructing a 4D Attention Mask for R-SWA training.
        shape: [batch_size, 1, seq_len, seq_len]

        Rules: Position i can attend to:
            1. prefill region [0, prefill_len) with causal constraint
            2. sliding window [i-W+1, i] after prefill
        Each batch item has its own prefill_len via prefill_lens: [batch_size].
        """
        min_val = torch.finfo(dtype).min
        batch_size = prefill_lens.shape[0]

        row = torch.arange(seq_len, device=device).view(1, seq_len, 1)  # [1, seq, 1]
        col = torch.arange(seq_len, device=device).view(1, 1, seq_len)  # [1, 1, seq]
        pl = prefill_lens.view(batch_size, 1, 1)  # [B, 1, 1]

        causal = col <= row
        prefill_vis = (col < pl) & causal
        window_vis = (col >= pl) & (col > row - window_size) & causal

        mask = torch.full((batch_size, seq_len, seq_len), min_val, dtype=dtype, device=device)
        mask[prefill_vis | window_vis] = 0

        return mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

    @staticmethod
    def _apply_rswa_training_patch(model, ring_window: int):
        """
        Monkey patch UnlimitedOCRForCausalLM instance forward.
        Replace attention_mask with R-SWA mask during training.
        """
        logger = get_logger()

        _original_forward = model.__class__.forward
        _make_mask = UnlimitedOCRLoader._make_rswa_causal_mask
        _CACHE_MAX = 8
        _mask_cache = {}

        _sig = inspect.signature(_original_forward)
        _param_list = list(_sig.parameters.keys())

        _attn_mask_pos = (_param_list.index('attention_mask') - 1 if 'attention_mask' in _param_list else None)

        def _get_arg(name, args, kwargs):
            if name in kwargs:
                return kwargs[name]
            try:
                idx = _param_list.index(name) - 1
                if 0 <= idx < len(args):
                    return args[idx]
            except ValueError:
                pass
            return None

        def _patched_forward(self, *args, **kwargs):
            if not self.training:
                return _original_forward(self, *args, **kwargs)

            input_ids = _get_arg('input_ids', args, kwargs)
            inputs_embeds = _get_arg('inputs_embeds', args, kwargs)
            images_seq_mask = _get_arg('images_seq_mask', args, kwargs)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                device = input_ids.device
            elif inputs_embeds is not None:
                batch_size = inputs_embeds.shape[0]
                seq_len = inputs_embeds.shape[1]
                device = inputs_embeds.device
            else:
                return _original_forward(self, *args, **kwargs)

            # Vectorized per-sample prefill_len computation.
            # images_seq_mask: [batch, seq_len] bool
            # Multiply by 1-based position indices, then take max per sample
            # to get the last image token position + 1.
            if images_seq_mask is not None:
                seq_indices = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)
                prefill_lens = (images_seq_mask.to(dtype=torch.long, device=device)
                                * seq_indices).max(dim=1)[0]  # [batch_size]
            else:
                prefill_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

            if prefill_lens.max().item() == 0:
                return _original_forward(self, *args, **kwargs)

            dtype = next(self.parameters()).dtype

            cache_key = (seq_len, tuple(prefill_lens.tolist()), dtype, str(device))
            if cache_key not in _mask_cache:
                if len(_mask_cache) >= _CACHE_MAX:
                    _mask_cache.pop(next(iter(_mask_cache)))
                _mask_cache[cache_key] = _make_mask(
                    seq_len=seq_len,
                    prefill_lens=prefill_lens,
                    window_size=ring_window,
                    dtype=dtype,
                    device=device,
                )

            rswa_mask = _mask_cache[cache_key]

            if (_attn_mask_pos is not None and _attn_mask_pos < len(args) and 'attention_mask' not in kwargs):
                args = list(args)
                args[_attn_mask_pos] = rswa_mask
                args = tuple(args)
            else:
                kwargs['attention_mask'] = rswa_mask

            return _original_forward(self, *args, **kwargs)

        model.forward = types.MethodType(_patched_forward, model)
        model._rswa_training_patched = True
        logger.info('[UnlimitedOCR] R-SWA training mask patch applied: ring_window=%d', ring_window)

    @staticmethod
    def _apply_multi_gpu_patch():
        """
        Fixed two bugs affecting `UnlimitedOCRModel` in multi-GPU scenarios using `device_map='auto'`:

        Bug 1 - Device mismatch in `torch.cat`:
            `image_newline` and `view_seperator` are `nn.Parameter`s;
            under `device_map='auto'`, their device placement might not align
            with the image features.

        Bug 2 - Device mismatch in `masked_scatter_`:
            Hard-coded `.cuda()` usage caused a conflict where
            `images_in_this_batch` resided on the projector's device (e.g., `cuda:7`),
            while `inputs_embeds` resided on the device hosting `embed_tokens`
            (e.g., `cuda:0`).

        Fix strategy: Temporarily replace `torch.cat` and
        `torch.Tensor.masked_scatter_` during the forward pass to handle device
        placement automatically, then restore the original methods after execution.
        """
        modeling_module = None
        for mod_name, mod in sys.modules.items():
            if 'modeling_unlimitedocr' in mod_name:
                modeling_module = mod
                break

        if modeling_module is None:
            return False

        UnlimitedOCRModel = getattr(modeling_module, 'UnlimitedOCRModel', None)
        if UnlimitedOCRModel is None:
            return False

        # Avoid redundant patching
        if getattr(UnlimitedOCRModel, '_swift_multi_gpu_patched', False):
            return True

        _original_forward = UnlimitedOCRModel.forward

        def _patched_forward(self, *args, **kwargs):
            _orig_cat = torch.cat
            _orig_masked_scatter_ = torch.Tensor.masked_scatter_

            def _safe_cat(tensors, dim=0, **cat_kwargs):
                # Using the device of the first tensor as reference
                ref_device = None
                for t in tensors:
                    if isinstance(t, torch.Tensor):
                        ref_device = t.device
                        break
                if ref_device is None:
                    return _orig_cat(tensors, dim, **cat_kwargs)
                aligned = [
                    t.to(ref_device) if isinstance(t, torch.Tensor) and t.device != ref_device else t for t in tensors
                ]
                return _orig_cat(aligned, dim, **cat_kwargs)

            def _safe_masked_scatter_(tensor_self, mask, source):
                # Use the device of tensor_self as reference
                dev = tensor_self.device
                if mask.device != dev:
                    mask = mask.to(dev)
                if source.device != dev:
                    source = source.to(dev)
                return _orig_masked_scatter_(tensor_self, mask, source)

            # Replace in both module namespace and global scope
            modeling_module.torch.cat = _safe_cat
            torch.cat = _safe_cat
            torch.Tensor.masked_scatter_ = _safe_masked_scatter_
            try:
                return _original_forward(self, *args, **kwargs)
            finally:
                # Restore to avoid contaminating other modules
                modeling_module.torch.cat = _orig_cat
                torch.cat = _orig_cat
                torch.Tensor.masked_scatter_ = _orig_masked_scatter_

        UnlimitedOCRModel.forward = _patched_forward
        UnlimitedOCRModel._swift_multi_gpu_patched = True
        return True

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        logger = get_logger()

        self.auto_model_cls = self.auto_model_cls or AutoModel
        model = super(DeepseekOCRLoader, self).get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.sam_model)
        patch_output_to_input_device(getattr(model.model, self.visual_name))
        patch_output_to_input_device(model.model.projector)
        patch_output_to_input_device(model.model)

        _orig_sw = (getattr(model.config, 'sliding_window_size', None) or getattr(model.config, 'sliding_window', None))
        if _orig_sw is not None:
            model.config._ring_window = _orig_sw
            model.config.sliding_window = None
            logger.info('[UnlimitedOCR] R-SWA enabled: ring_window=%d', _orig_sw)

            UnlimitedOCRLoader._apply_rswa_training_patch(model, _orig_sw)
        else:
            logger.warning('[UnlimitedOCR] sliding_window config not found, R-SWA may not work.')

        n_devices = len(set(str(p.device) for p in model.parameters() if p.device.type == 'cuda'))
        if n_devices > 1:
            if UnlimitedOCRLoader._apply_multi_gpu_patch():
                logger.info('[UnlimitedOCR] Multi-GPU patch applied (%d GPUs).', n_devices)
            else:
                logger.warning('[UnlimitedOCR] Multi-GPU deployment failed to apply patch.'
                               'If an inference error occurs, please check whether'
                               ' `modeling_unlimitedocr` has been loaded correctly.')

        return model


register_model(
    ModelMeta(
        MLLMModelType.unlimited_ocr,
        [
            ModelGroup([
                Model('PaddlePaddle/Unlimited-OCR', 'PaddlePaddle/Unlimited-OCR'),
            ]),
        ],
        UnlimitedOCRLoader,
        template=TemplateType.unlimited_ocr,
        model_arch=ModelArch.unlimited_ocr,
        architectures=['UnlimitedOCRForCausalLM'],
        requires=['transformers==4.46.3', 'easydict'],
        tags=['vision'],
    ))
