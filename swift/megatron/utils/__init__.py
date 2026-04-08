# Copyright (c) ModelScope Contributors. All rights reserved.

from .convert_utils import test_convert_precision
from .megatron_lm_utils import (disable_forward_pre_hook, enable_forward_pre_hook, get_optimizer_param_scheduler,
                                init_persistent_async_worker, initialize_megatron, initialize_tp_communicators,
                                load_mcore_checkpoint, maybe_finalize_async_save, save_mcore_checkpoint,
                                should_disable_forward_pre_hook, warmup_jit_function, wrap_model)
from .parallel_utils import logical_and_across_model_parallel_group, reduce_max_stat_across_model_parallel_group
from .patcher import patch_merge_fn, patch_torch_dist_shard
from .router_replay_utils import (RouterReplayHelper, apply_router_replay_patch, get_local_topk_idx_for_current_rank,
                                  get_router_replay_data, set_router_replay_data)
from .utils import forward_step_helper, get_padding_to, prepare_mcore_model
