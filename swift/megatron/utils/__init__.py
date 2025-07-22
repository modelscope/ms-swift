# Copyright (c) Alibaba, Inc. and its affiliates.

from .convert import convert_hf2mcore, convert_mcore2hf
from .patcher import patch_megatron_tokenizer
from .utils import (adapter_state_dict_context, copy_original_module_weight, prepare_mcore_model,
                    tuners_sharded_state_dict)
