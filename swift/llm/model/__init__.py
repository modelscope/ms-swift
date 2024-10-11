# Copyright (c) Alibaba, Inc. and its affiliates.
from .constant import LLMModelType, MLLMModelType, ModelType
from .register import (MODEL_MAPPING, Model, ModelGroup, TemplateGroup, fix_do_sample_warning, get_default_device_map,
                       get_default_torch_dtype, get_model_tokenizer, get_model_tokenizer_from_local,
                       get_model_tokenizer_with_flash_attn)
from .utils import HfConfigFactory, safe_snapshot_download


def _register_files():
    from . import qwen
    from . import llama
    # TODO
    # from . import model


_register_files()
