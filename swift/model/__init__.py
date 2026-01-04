# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers.utils import is_torch_npu_available

from . import models
from .constant import LLMModelType, MLLMModelType, ModelType
from .model_arch import MODEL_ARCH_MAPPING, ModelArch, ModelKeys, MultiModelKeys, get_model_arch, register_model_arch
from .model_meta import Model, ModelGroup, ModelInfo, ModelMeta, get_matched_model_meta, get_model_name
from .register import (MODEL_MAPPING, ModelLoader, fix_do_sample_warning, get_default_device_map, get_model,
                       get_model_info_meta, get_model_tokenizer, get_model_tokenizer_multimodal, load_by_unsloth,
                       register_model)
from .utils import get_default_torch_dtype, get_llm_model, git_clone_github, safe_snapshot_download

if is_torch_npu_available():
    from . import npu_patcher
