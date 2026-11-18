"""
Patches for ms-swift training (HY4 Flash).

Registers the hy_v4_flash_vl model type and template for ms-swift, including:
    1. Model & Template registration: Register hy_v4_flash_vl model_type
       with is_multimodal=True and VL architecture.
    2. Multimodal data processing (image/video encoding).
    3. Training optimizations (gradient checkpointing, grad norm, etc.).

Usage:
    swift sft --custom_register_path hy_v4_flash_swift_patches.py --model /path/to/hy4_flash_ckpt ...
"""

# Source: Tencent-Hunyuan/Hy4-flash/finetune/ms_swift_support/hy_v4_flash_swift_patches.py
# Kept close to the vendor implementation and loaded through `--custom_register_path`.
# Requires a transformers build that ships `hy_v4_flash` (huggingface/transformers#50903).

import os
import gc
import json as _json
import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Patch 1: VL Model & Template Registration
#
# Register HY4-Flash-VL as a multimodal model in ms-swift. The VL model uses
# HYV4FlashForConditionalGeneration and needs image/video processing support.
# ============================================================================

from swift.model import register_model, ModelMeta, ModelGroup, Model
from swift.model.register import ModelLoader
from swift.model.model_arch import MultiModelKeys, register_model_arch
from swift.template import register_template, TemplateMeta

# Register HY4-Flash-VL model architecture mapping.
# HYV4FlashForConditionalGeneration has:
#   self.model = HYV4FlashModel (contains self.vision_tower and self.language_model)
#   self.lm_head
HYV4_FLASH_VL_ARCH_NAME = 'hy_v4_flash_vl'
register_model_arch(
    MultiModelKeys(
        HYV4_FLASH_VL_ARCH_NAME,
        language_model=['model.language_model', 'lm_head'],
        vision_tower='model.vision_tower',
        aligner=[],
    ),
    exist_ok=True,
)


# Custom ModelLoader for HY4-Flash-VL
class HYV4FlashVLModelLoader(ModelLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or HYV4FlashForConditionalGeneration
        return super().get_model(model_dir, config, processor, model_kwargs)

# ---------------------------------------------------------------------------
# HYV4FlashVLTemplate: multimodal template for HY4-Flash-VL
# ---------------------------------------------------------------------------
from swift.template.base import Template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.utils import Context, findall
from typing import List, Literal, Optional
from PIL import Image as PILImage

class HYV4FlashVLTemplate(Template):
    """Multimodal template for HY4-Flash-VL models in ms-swift."""

    # HY4-Flash-VL image/video token IDs (from config.json)
    image_token_id = 120120
    video_token_id = 120124
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            return [[-100]]
        else:
            return [[-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        # Process images
        if inputs.images:
            images = []
            for img in inputs.images:
                if isinstance(img, str):
                    images.append(PILImage.open(img).convert('RGB'))
                else:
                    images.append(img)

            image_processor = processor.image_processor
            image_inputs = image_processor(images=images, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            merge_size = image_processor.merge_size

            idx_list = findall(input_ids, -100)

            def _get_new_tokens(i):
                grid_h, grid_w = image_grid_thw[i][-2:]
                patch_h = grid_h // merge_size
                patch_w = grid_w // merge_size
                num_tokens = int(patch_h * (patch_w + 1))
                return [self.image_token_id] * num_tokens

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens)
            encoded.update(image_inputs)

        # Process videos
        if inputs.videos:
            video_processor = getattr(processor, 'video_processor', processor.image_processor)
            video_inputs = video_processor(videos=inputs.videos, return_tensors='pt')
            video_grid_thw = video_inputs['video_grid_thw']
            merge_size = processor.image_processor.merge_size

            idx_list = findall(input_ids, -200)

            def _get_new_video_tokens(i):
                grid_t, grid_h, grid_w = video_grid_thw[i]
                patch_h = grid_h // merge_size
                patch_w = grid_w // merge_size
                num_tokens = int(grid_t * patch_h * (patch_w + 1))
                return [self.video_token_id] * num_tokens

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_video_tokens)
            encoded.update(video_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        if loss_scale is not None:
            encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)

        input_ids = res.get('input_ids')
        if input_ids is not None:
            image_grid_thw = res.get('image_grid_thw', None)
            video_grid_thw = res.get('video_grid_thw', None)

            if image_grid_thw is not None or video_grid_thw is not None:
                mm_token_type_ids = torch.zeros_like(input_ids)
                image_token_id = getattr(self, 'image_token_id', 120120)
                video_token_id = getattr(self, 'video_token_id', 120124)
                mm_token_type_ids[input_ids == image_token_id] = 1
                mm_token_type_ids[input_ids == video_token_id] = 2

                try:
                    base_model = self.get_base_model(self._get_model())
                    if hasattr(base_model, 'get_rope_index'):
                        position_ids, _ = base_model.get_rope_index(
                            input_ids,
                            mm_token_type_ids=mm_token_type_ids,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            attention_mask=res.get('attention_mask'),
                        )
                        res['position_ids'] = position_ids
                except Exception as e:
                    logger.warning("[HYV4FlashVLTemplate] Failed to compute MRoPE position_ids: %s", e)

                res['mm_token_type_ids'] = mm_token_type_ids

        return res


# Register hy_v4_flash_vl template with HYV4FlashVLTemplate for multimodal support
# Token format: <｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>{content}<｜hy_end:opensource｜>
# default_system injects the reasoning_mode system prompt (same as LLaMA Factory).
# non_thinking_prefix inserts an empty <think></think> block before assistant responses
# in no_think mode, matching the format the model was trained with.
register_template(
    TemplateMeta(
        template_type='hy_v4_flash_vl',
        prefix=[],
        system_prefix=['<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>{{SYSTEM}}<｜hy_end:opensource｜>'],
        prompt=['<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{{QUERY}}<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>'],
        chat_sep=['<｜hy_end:opensource｜>'],
        suffix=['<｜hy_end:opensource｜>'],
        default_system='<｜reasoning_mode:opensource｜>reasoning_effort:no_think',
        non_thinking_prefix='<think:opensource></think:opensource>',
        template_cls=HYV4FlashVLTemplate,
    ),
    exist_ok=True,
)

# Also register the LLM-only template
register_template(
    TemplateMeta(
        template_type='hy_v4_flash',
        prefix=[],
        system_prefix=['<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>{{SYSTEM}}<｜hy_end:opensource｜>'],
        prompt=['<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{{QUERY}}<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>'],
        chat_sep=['<｜hy_end:opensource｜>'],
        suffix=['<｜hy_end:opensource｜>'],
        default_system='<｜reasoning_mode:opensource｜>reasoning_effort:no_think',
        non_thinking_prefix='<think:opensource></think:opensource>',
    ),
    exist_ok=True,
)

# Register hy_v4_flash_vl model (multimodal)
register_model(
    ModelMeta(
        model_type='hy_v4_flash_vl',
        model_groups=[
            ModelGroup([
                Model('Tencent-Hunyuan/Hy4-Flash-VL',
                      'Tencent-Hunyuan/Hy4-Flash-VL'),
            ]),
        ],
        loader=HYV4FlashVLModelLoader,
        template='hy_v4_flash_vl',
        model_arch=HYV4_FLASH_VL_ARCH_NAME,
        architectures=['HYV4FlashForConditionalGeneration'],
        is_multimodal=True,
    ),
    exist_ok=True,
)

# Also register the LLM-only model
register_model(
    ModelMeta(
        model_type='hy_v4_flash',
        model_groups=[
            ModelGroup([
                Model('Tencent-Hunyuan/Hy4-Flash',
                      'Tencent-Hunyuan/Hy4-Flash'),
            ]),
        ],
        template='hy_v4_flash',
        architectures=['HYV4FlashForCausalLM'],
        is_multimodal=False,
    ),
    exist_ok=True,
)

logger.info(
    "[HYV4-Flash Patch 1] Model types 'hy_v4_flash_vl' (multimodal) and 'hy_v4_flash' (text-only) "
    "registered in ms-swift."
)

# ============================================================================
# Patch 2: Skip grad norm computation for DeepSpeed ZeRO-3
# ============================================================================

def _apply_skip_grad_norm_patch():
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

        def _skip_get_norm_groups(self):
            return [torch.tensor(0.0)]

        DeepSpeedZeroOptimizer_Stage3._get_norm_groups = _skip_get_norm_groups
        logger.info("[HYV4-Flash Patch 2] Patched grad norm skip.")
    except ImportError:
        logger.info("[HYV4-Flash Patch 2] DeepSpeed not available, skipping.")

# ============================================================================
# Patch 3: Memory-efficient model loading for ZeRO-3 and FSDP1
# ============================================================================

def _apply_shard_loading_patch():
    import sys
    from transformers import AutoConfig, PreTrainedModel

    _real_orig_from_pretrained = PreTrainedModel.from_pretrained.__func__

    def _disable_engram(config):
        """Disable engram layers to avoid lookup_table size mismatch.

        The checkpoint's engram lookup_table buffers may have a different size
        (e.g. 120818) than config.text_config.vocab_size (120832). Setting
        engram_layer_ids=None prevents the model from creating engram layers
        entirely, avoiding the RuntimeError during weight loading.
        """
        text_config = getattr(config, 'text_config', config)
        old_ids = getattr(text_config, 'engram_layer_ids', None)
        if old_ids is not None:
            text_config.engram_layer_ids = None
            print(f"[HYV4-Flash Patch 3] _disable_engram: set engram_layer_ids=None "
                  f"(was {old_ids})", flush=True)

    def _disable_router_logits_if_needed(model):
        if hasattr(model, 'config') and getattr(model.config, 'output_router_logits', False):
            model.config.output_router_logits = False
            logger.info("[HYV4-Flash Patch 3] Disabled output_router_logits.")
        return model

    def _is_fsdp_requested():
        accel_fsdp = str(os.environ.get("ACCELERATE_USE_FSDP", "")).lower()
        if accel_fsdp in {"1", "true", "yes"}:
            return True
        return "--fsdp" in sys.argv

    def _build_meta_model_for_fsdp(cls, model_path, kwargs):
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

        init_kwargs = {}
        torch_dtype = kwargs.get("torch_dtype", None)
        if torch_dtype is not None:
            init_kwargs["torch_dtype"] = torch_dtype
        if "attn_implementation" in kwargs:
            init_kwargs["attn_implementation"] = kwargs["attn_implementation"]
        if "experts_implementation" in kwargs:
            init_kwargs["experts_implementation"] = kwargs["experts_implementation"]

        with torch.device("meta"):
            model = cls._from_config(config, **init_kwargs)
        return _disable_router_logits_if_needed(model)

    def _fsdp_safe_load(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("low_cpu_mem_usage", True)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            logger.info("[HYV4-Flash Patch 3] FSDP: local_rank=%d, meta device.", local_rank)
            return _build_meta_model_for_fsdp(cls, pretrained_model_name_or_path, kwargs)

        logger.info("[HYV4-Flash Patch 3] FSDP: local_rank=0, loading to CPU.")
        kwargs["device_map"] = {"": "cpu"}
        model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
        return _disable_router_logits_if_needed(model)

    def _ensure_zero3_config_and_load(cls, pretrained_model_name_or_path, *args, **kwargs):
        model_path = pretrained_model_name_or_path

        # Disable engram layers before model instantiation to avoid
        # lookup_table buffer size mismatch (ckpt=120818 vs config=120832).
        config = kwargs.get('config')
        if config is not None:
            _disable_engram(config)

        if not (isinstance(model_path, str) and os.path.isdir(model_path)):
            return _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        index_file = os.path.join(model_path, "model.safetensors.index.json")
        single_file = os.path.join(model_path, "model.safetensors")
        if not (os.path.isfile(index_file) or os.path.isfile(single_file)):
            return _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        if _is_fsdp_requested():
            return _fsdp_safe_load(cls, pretrained_model_name_or_path, *args, **kwargs)

        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        ds_config_path = os.environ.get("DEEPSPEED_CONFIG_FILE", None)
        if ds_config_path is None:
            ds_config_path = os.environ.get("DEEPSPEED_CONFIG", None)

        if ds_config_path is None:
            for i, arg in enumerate(sys.argv):
                if arg == '--deepspeed' and i + 1 < len(sys.argv):
                    ds_config_path = sys.argv[i + 1]
                    break

        if ds_config_path is None or not os.path.isfile(ds_config_path):
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        with open(ds_config_path, "r") as f:
            ds_config = _json.load(f)

        zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
        if zero_stage != 3:
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        _ds_config_obj = HfDeepSpeedConfig(ds_config_path)

        model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
        return _disable_router_logits_if_needed(model)

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return _ensure_zero3_config_and_load(cls, pretrained_model_name_or_path, *args, **kwargs)

    PreTrainedModel.from_pretrained = _patched_from_pretrained
    logger.info("[HYV4-Flash Patch 3] Loading patch applied.")

# ============================================================================
# Optional Patch: LoRA z3_leaf marking
# ============================================================================

def apply_lora_z3_leaf_patch(model):
    try:
        from deepspeed.utils import set_z3_leaf_module
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError:
        return

    z3_leaf_count = 0
    for module in model.modules():
        if isinstance(module, LoraLinear):
            set_z3_leaf_module(module, True)
            z3_leaf_count += 1
    logger.info("[HYV4-Flash] Marked %d LoraLinear modules with _z3_leaf=True.", z3_leaf_count)

# ============================================================================
# Patch 4: Fix logging_dir compatibility
# ============================================================================

def _apply_logging_dir_patch():
    try:
        from swift.arguments.sft_args import SftArguments

        _orig_add_version = SftArguments._add_version

        def _patched_add_version(self):
            if not hasattr(self, 'logging_dir'):
                self.logging_dir = None
            if not hasattr(self, 'run_name'):
                self.run_name = None
            _orig_add_version(self)

        SftArguments._add_version = _patched_add_version
        logger.info("[HYV4-Flash Patch 4] logging_dir compatibility patch applied.")
    except (ImportError, AttributeError) as e:
        logger.info("[HYV4-Flash Patch 4] Could not apply: %s", e)

# ============================================================================
# Patch 5: FSDP1 dtype + mixed precision guard
# ============================================================================

def _apply_fsdp_dtype_patch():
    try:
        from transformers import Trainer

        _orig_prepare_for_training = Trainer._prepare_for_training

        def _patched_prepare_for_training(self, *args, **kwargs):
            if getattr(self.args, 'fsdp', False):
                dtype_counts = {}
                for p in self.model.parameters():
                    dt = str(p.dtype)
                    dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

                if len(dtype_counts) > 1:
                    logger.info("[HYV4-Flash Patch 5] Mixed dtypes: %s. Casting to bf16.", dtype_counts)
                    for p in self.model.parameters():
                        if p.dtype != torch.bfloat16 and p.dtype.is_floating_point:
                            p.data = p.data.to(torch.bfloat16)

                try:
                    if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'state'):
                        self.accelerator.state._mixed_precision = 'no'
                        fsdp_plugin = getattr(self.accelerator.state, 'fsdp_plugin', None)
                        if fsdp_plugin is not None:
                            if hasattr(fsdp_plugin, 'mixed_precision_policy'):
                                fsdp_plugin.mixed_precision_policy = None
                            if hasattr(fsdp_plugin, 'kwargs') and isinstance(fsdp_plugin.kwargs, dict):
                                fsdp_plugin.kwargs.pop('mixed_precision', None)
                    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
                except Exception as e:
                    logger.warning("[HYV4-Flash Patch 5] Failed to disable mixed precision: %s", e)

            # ---- Fix PEFT + FSDP _no_split_modules mismatch ----
            # PEFT's fsdp_auto_wrap_policy iterates _no_split_modules and calls
            # get_module_class_from_name() for each class name. If a class is
            # listed but not instantiated in the model (e.g. HYV4FlashEngramLayer
            # when engram is disabled), the lookup returns None and PEFT raises:
            #   "Could not find the transformer layer class to wrap in the model."
            # Fix: collect all actual module class names and prune _no_split_modules.
            try:
                actual_cls_names = {m.__class__.__name__ for m in self.model.modules()}
                for sub in self.model.modules():
                    if hasattr(sub, '_no_split_modules') and sub._no_split_modules:
                        original = list(sub._no_split_modules)
                        pruned = [c for c in original if c in actual_cls_names]
                        if len(pruned) < len(original):
                            removed = set(original) - set(pruned)
                            sub._no_split_modules = pruned
                            logger.info(
                                "[HYV4-Flash Patch 5] Pruned _no_split_modules: removed %s "
                                "(not found in model). Remaining: %s",
                                removed, pruned
                            )
            except Exception as e:
                logger.warning("[HYV4-Flash Patch 5] Failed to prune _no_split_modules: %s", e)

            return _orig_prepare_for_training(self, *args, **kwargs)

        Trainer._prepare_for_training = _patched_prepare_for_training
        logger.info("[HYV4-Flash Patch 5] FSDP1 dtype guard applied.")
    except (ImportError, AttributeError) as e:
        logger.info("[HYV4-Flash Patch 5] Could not apply: %s", e)

# ============================================================================
# Patch 6: Disable _compute_acc during training
# ============================================================================

def _apply_disable_compute_acc_patch():
    try:
        from swift.trainers.mixin import SwiftMixin

        def _noop_compute_acc(self, outputs, labels, cu_seqlens=None):
            return

        SwiftMixin._compute_acc = _noop_compute_acc
        logger.info("[HYV4-Flash Patch 6] Disabled _compute_acc.")
    except (ImportError, AttributeError) as e:
        logger.warning("[HYV4-Flash Patch 6] Could not apply: %s", e)

# ============================================================================
# Patch 7: Freeze ViT (handled natively by ms-swift)
# ============================================================================

def _apply_freeze_vit_patch():
    logger.info("[HYV4-Flash Patch 7] ViT freeze handled by ms-swift native freeze_vit + model_arch.")

# ============================================================================
# Patch 8: tokenizer → processing_class compatibility
# ============================================================================

def _apply_tokenizer_to_processing_class_patch():
    try:
        from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer

        _original_hf_seq2seq_init = HfSeq2SeqTrainer.__init__

        def _patched_hf_seq2seq_init(self, *args, **kwargs):
            if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
                kwargs['processing_class'] = kwargs.pop('tokenizer')
            return _original_hf_seq2seq_init(self, *args, **kwargs)

        HfSeq2SeqTrainer.__init__ = _patched_hf_seq2seq_init
        logger.info("[HYV4-Flash Patch 8] tokenizer→processing_class patch applied.")
    except (ImportError, AttributeError) as e:
        logger.info("[HYV4-Flash Patch 8] Could not apply: %s", e)

# ============================================================================
# Patch 9: Make CUDA pin_memory fault-tolerant
# ============================================================================

def _apply_safe_pin_memory_patch():
    try:
        from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator

        _orig_pin_memory = CUDA_Accelerator.pin_memory

        def _safe_pin_memory(self, tensor):
            try:
                return _orig_pin_memory(self, tensor)
            except Exception as e:
                import torch
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(f"[HYV4-Flash Patch 9] WARNING: pin_memory() failed on rank {rank}: {e}. "
                      f"Falling back to unpinned CPU tensor.", flush=True)
                return tensor

        CUDA_Accelerator.pin_memory = _safe_pin_memory
        logger.info("[HYV4-Flash Patch 9] pin_memory fault-tolerant fallback applied.")
    except ImportError:
        pass


# ============================================================================
# Auto-apply patches on import
# ============================================================================

_apply_safe_pin_memory_patch()
_apply_skip_grad_norm_patch()
_apply_shard_loading_patch()
_apply_logging_dir_patch()
_apply_fsdp_dtype_patch()
_apply_disable_compute_acc_patch()
_apply_freeze_vit_patch()
_apply_tokenizer_to_processing_class_patch()

logger.info("[HYV4-Flash] All ms-swift patches loaded successfully.")
