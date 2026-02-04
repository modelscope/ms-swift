# Copyright (c) ModelScope Contributors. All rights reserved.

from .config import convert_hf_config
from .convert_utils import test_convert_precision
from .megatron_lm_utils import initialize_megatron, load_mcore_checkpoint, save_mcore_checkpoint
from .patcher import patch_merge_fn, patch_torch_dist_shard
from .utils import (MegatronTrainerState, adapter_state_dict_context, copy_original_module_weight, forward_step_helper,
                    get_local_layer_specs, get_padding_to, prepare_mcore_model, tuners_sharded_state_dict)
