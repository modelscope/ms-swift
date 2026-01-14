# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers.utils import is_torch_npu_available

from . import models
from .constant import LLMModelType, MLLMModelType, ModelType
from .model_arch import MODEL_ARCH_MAPPING, ModelArch, ModelKeys, MultiModelKeys, get_model_arch, register_model_arch
from .model_meta import Model, ModelGroup, ModelInfo, ModelMeta, get_matched_model_meta, get_model_name
from .patcher import get_lm_head_model
from .register import (MODEL_MAPPING, ModelLoader, fix_do_sample_warning, get_default_device_map, get_model_info_meta,
                       get_model_list, get_model_processor, get_processor, load_by_unsloth, register_model)
from .utils import get_ckpt_dir, get_default_torch_dtype, get_llm_model, save_checkpoint

if is_torch_npu_available():
    from . import npu_patcher
