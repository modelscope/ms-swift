# Copyright (c) Alibaba, Inc. and its affiliates.

from .env import (get_dist_setting, get_pai_tensorboard_dir, is_ddp_plus_mp, is_deepspeed_enabled, is_dist, is_dist_ta,
                  is_local_master, is_master, is_mp, is_pai_training_job, torchacc_trim_graph, use_hf_hub, use_torchacc)
from .import_utils import (is_liger_available, is_lmdeploy_available, is_megatron_available, is_merge_kit_available,
                           is_unsloth_available, is_vllm_available, is_xtuner_available)
from .io_utils import append_to_jsonl, download_ms_file, read_from_jsonl, write_to_jsonl
from .logger import get_logger
from .metric import compute_acc, compute_acc_metrics, compute_nlg_metrics, preprocess_logits_for_acc
from .np_utils import get_seed, stat_array, transform_jsonl_to_df
from .tb_utils import TB_COLOR, TB_COLOR_SMOOTH, plot_images, read_tensorboard_file, tensorboard_smoothing
from .torch_utils import (activate_model_parameters, broadcast_string, find_all_linears, find_embedding,
                          freeze_model_parameters, get_model_parameter_info, is_on_same_device, safe_ddp_context,
                          show_layers, time_synchronize)
from .utils import (add_version_to_work_dir, check_json_format, dataclass_to_dict, deep_getattr, get_env_args,
                    lower_bound, parse_args, read_multi_line, seed_everything, subprocess_run, test_time, upper_bound)
