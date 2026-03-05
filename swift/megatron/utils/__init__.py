# Copyright (c) ModelScope Contributors. All rights reserved.

from .convert_utils import test_convert_precision
from .megatron_lm_utils import (disable_forward_pre_hook, enable_forward_pre_hook, get_optimizer_param_scheduler,
                                init_persistent_async_worker, initialize_megatron, initialize_tp_communicators,
                                load_mcore_checkpoint, maybe_finalize_async_save, save_mcore_checkpoint,
                                set_random_seed, should_disable_forward_pre_hook, unwrap_model, warmup_jit_function,
                                wrap_model)
from .parallel_utils import (logical_and_across_model_parallel_group, reduce_max_stat_across_model_parallel_group,
                             split_cp_inputs)
from .patcher import patch_merge_fn, patch_torch_dist_shard
from .utils import (copy_original_module_weight, forward_step_helper, get_local_layer_specs, get_padding_to,
                    prepare_mcore_model, tuners_sharded_state_dict)
