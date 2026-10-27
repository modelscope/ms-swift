# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Type

UNSUPPORTED_SAMPLER_FIELDS: Tuple[str, ...] = (
    'lmdeploy_cache_max_entry_count', 'lmdeploy_quant_policy', 'lmdeploy_session_len', 'lmdeploy_tp',
    'lmdeploy_vision_batch_size')
# v4-only knobs: write_batch_size/to_ollama were superseded, the rest were removed from the Configs
# (reward selectors merged into --orm/--prm, plugins onto --external_plugins, tuner_backend onto --backend).
REMOVED_OPTION_FIELDS: Tuple[str, ...] = (
    'ignore_args_error', 'use_swift_lora', 'write_batch_size', 'to_ollama', 'orm_model', 'prm_model',
    'reward_model_plugin', 'custom_register_path', 'tuner_backend',
    # device_map is retired: the runtime places the model through its own parallel/accelerate strategy.
    'device_map',
    # use_vllm is retired: a rollout always rides a weight-syncable sampler engine (--rollout_sampler).
    'use_vllm',
    # Template knobs not consumed: loss scale is --loss_scale, thinking-block handling --enable_thinking.
    'is_binary_loss_scale', 'template_backend', 'preserve_thinking', 'disable_ignore_empty_think',
    # No dev consumer: the eval budget is max_new_tokens/num_beams (--eval_generation_config for
    # in-training eval), the attention-backend pick folds onto --attn_impl, the rest are builtin.
    'generation_max_length', 'generation_num_beams', 'norm_bbox', 'log_interval',
    # AdaLoRA is retired pending its rank-allocation schedule getting a runtime hook: tuner='adalora'
    # silently degraded to a fixed-rank LoRA, so both the tuner value and its schedule knobs refuse.
    'adalora_target_r', 'adalora_init_r', 'adalora_tinit', 'adalora_tfinal', 'adalora_deltaT',
    'adalora_beta1', 'adalora_beta2', 'adalora_orth_reg_weight',
    # The engine's LoRA machinery is owned by the adapters themselves: enable is automatic when
    # adapters are passed, and the rank budget is measured from each adapter_config.json.
    'vllm_enable_lora', 'vllm_max_lora_rank')
# Per-field replacement text where the migration path is specific enough to name.
REMOVED_OPTION_REPLACEMENTS: Dict[str, str] = {
    'vllm_enable_lora': 'LoRA support is enabled automatically whenever adapters are passed (--adapters'
                        ' / --orm_adapters); drop this flag',
    'vllm_max_lora_rank': 'the rank budget is measured from each adapter\'s own adapter_config.json; '
                          'drop this flag',
    'orm_model': 'pass the reward-model id as an item of --orm instead',
    'prm_model': 'pass the reward-model id as an item of --prm instead',
    'reward_model': 'pass the reward-model id as an item of --orm instead',
    'reward_adapters': 'pass the LoRA through --orm_adapters (one entry per model id in --orm) instead',
    'reward_model_type': 'pass the override through --orm_model_type (one entry per model id) instead',
    'reward_model_revision': 'pin it through --orm_model_revision (one entry per model id) instead',
    'reward_template': 'pass the template through --orm_template (one entry per model id) instead',
    'reward_model_plugin': 'reward-model scoring is built in now; configure --orm/--prm directly',
    'custom_register_path': 'use --external_plugins (local file | local folder | hub id)',
    'tuner_backend': 'select unsloth with --backend unsloth instead (it covers LoRA and full-parameter runs)',
    'device_map': 'the runtime places the model across devices on its own; drop --device_map',
    'use_vllm': 'an online-RL rollout always uses a weight-syncable sampler engine; select it with '
                '--rollout_sampler (vllm/sglang) instead, or drop --use_vllm',
    'generation_max_length': 'use --max_new_tokens, or --eval_generation_config for in-training eval',
    'generation_num_beams': 'use --num_beams, or --eval_generation_config for in-training eval',
    'norm_bbox': 'bounding-box coordinate normalization follows the template; drop --norm_bbox',
    'log_interval': 'log and throughput cadence is builtin and not configurable; drop --log_interval',
    'adalora_target_r': 'adaptive-rank LoRA is retired; use plain --tuner lora (its rank is fixed by --lora_rank)',
    'merge_lora': 'this command merges the adapter implicitly; drop --merge_lora',
    'exist_ok': 'use --replace_if_exists to overwrite an existing output directory',
    'steps_per_generation': 'use --num_iterations to replay one rollout across several optimizer steps',
    'num_mini_batches': 'set --per_device_train_batch_size to control the mini-batch width',
    'num_generations_eval': 'set the per-prompt generation count inside --eval_generation_config instead',
    'prediction_loss_only': 'evaluation already computes the loss only; drop --prediction_loss_only',
    'do_eval': 'use --eval_strategy to enable evaluation',
    'eval_use_evalscope': 'in-training evaluation is configured through --eval_strategy and --eval_dataset',
    'label_names': 'the data pipeline always produces labels; drop --label_names',
    'max_epochs': 'use --num_train_epochs instead',
    'disable_tqdm': 'training progress is written to the log rather than a progress bar; drop --disable_tqdm',
    'check_model': 'the model is always checked on load; drop --check_model',
    'lora_dtype': 'the adapter dtype follows the base model; drop --lora_dtype',
    'eval_metric': 'metrics are configured on the optimizer status; drop --eval_metric',
    'average_tokens_across_devices': 'the loss is already averaged over real tokens across the whole global batch',
    'use_logits_to_keep': 'memory-efficient loss computation is always enabled',
    'is_binary_loss_scale': 'the loss scale is configured through --loss_scale; drop --is_binary_loss_scale',
    'template_backend': 'the template backend is picked automatically (jinja only as a fallback); drop --template_backend',
    'preserve_thinking': 'thinking-block handling is configured through --enable_thinking; drop --preserve_thinking',
    'disable_ignore_empty_think': 'empty-think handling is not configurable; drop --disable_ignore_empty_think',
    'seq_kd': 'sequence-level knowledge distillation is expressed through the GKD objective; drop --seq_kd',
}
# eval scores a local sampler or a --eval_url endpoint through the native runner, so the remote
# backend selector and the OpenCompass-only local-data toggle are obsolete.
EVAL_REMOTE_EVAL_FIELDS: Tuple[str, ...] = ('eval_backend', 'local_dataset')
# Commands that never decode text (export) or set decoding through one structured flag (eval's
# --eval_generation_config) don't consume these per-flag generation options.
GENERATION_OPTION_FIELDS: Tuple[str, ...] = (
    'temperature', 'top_p', 'top_k', 'num_beams', 'repetition_penalty', 'stop_words', 'stream', 'logprobs',
    'top_logprobs', 'max_new_tokens', 'structured_outputs_regex')
# Rollout knobs retired together with the gym environments and the pluggable multi-turn scheduler.
RLHF_REMOVED_ROLLOUT_FIELDS: Tuple[str, ...] = (
    'gym_env', 'use_gym_env', 'multi_turn_scheduler', 'completion_length_limit_scope')
# The reward_model family retired into the --orm/--prm channels: a reward model is a model-id item of
# the channel, its LoRA/type/revision/template ride the channel's companion lists on RewardConfig.
REWARD_MODEL_RETIRED_FIELDS: Tuple[str, ...] = (
    'reward_model', 'reward_adapters', 'reward_model_type', 'reward_model_revision', 'reward_template')
# RL rollout/training knobs parsed but never consumed: the round-based sample-reuse knobs, the batched
# weight-move/flattened sync buffers, and the weight-bridge offload between syncs.
RLHF_REMOVED_RL_KNOBS: Tuple[str, ...] = (
    'move_model_batches', 'enable_flattened_weight_sync', 'steps_per_generation', 'num_mini_batches',
    'local_rollout_forward_batch_size', 'num_sample_generations', 'missing_eos_penalty', 'offload_bridge')
# Eval scores a sampler through the benchmark runner, so there is no per-prompt eval generation to
# size; the budget is expressed through the structured --eval_generation_config instead.
RLHF_REMOVED_EVAL_FIELDS: Tuple[str, ...] = ('num_generations_eval',)
# HTTP serving knobs retired with the server rollout path: an RL run starts its sampler/teacher inside
# the job per its own parallel spec, so there is no user-facing server address to point at.
RLHF_REMOVED_SERVER_FIELDS: Tuple[str, ...] = (
    'teacher_model_server', 'teacher_tag_key', 'vllm_server_base_url', 'vllm_server_group_port',
    'vllm_server_host', 'vllm_server_pass_dataset', 'vllm_server_port', 'vllm_server_timeout')
# eval scores an in-process sampler and starts no HTTP deployment; the server knobs it once accepted
# through DeployConfig belong to `swift deploy`, not here.
EVAL_SERVING_FIELDS: Tuple[str, ...] = (
    'host', 'port', 'api_key', 'served_model_name', 'owned_by', 'ssl_certfile', 'ssl_keyfile', 'max_logprobs',
    'log_level', 'verbose')
UNSUPPORTED_DISTRIBUTED_FIELDS: Tuple[str, ...] = ('ddp_backend', 'ddp_timeout', 'device_groups', 'ray_exp_name')
SERVING_DISTRIBUTED_FIELDS: Tuple[str, ...] = UNSUPPORTED_DISTRIBUTED_FIELDS + ('use_ray',)
# Training consumes --ray_exp_name (twinkle.initialize) and --ddp_timeout (the strategy timeout); the
# backend is fixed by the device platform and the layout is derived, so those two stay refused.
TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS: Tuple[str, ...] = ('ddp_backend', 'device_groups')
TRAIN_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'optim_target_modules', 'log_on_each_node', 'include_num_input_tokens_seen', 'log_level', 'log_level_replica',
    'project', 'trackio_space_id', 'trackio_bucket_id', 'trackio_static_space_id', 'eval_do_concat_batches',
    'eval_use_gather_object', 'include_for_metrics', 'batch_eval_metrics', 'enable_jit_checkpoint',
    'restore_callback_states_from_checkpoint', 'use_cpu', 'parallelism_config', 'train_sampling_strategy',
    'length_column_name', 'debug', 'skip_memory_metrics', 'do_train', 'do_predict', 'sortish_sampler',
    # HF-Trainer/accelerate/DeepSpeed internals with no counterpart: graph compilation, the accelerate
    # config, OOM batch halving, the ZeRO-3 gather, eval offload/precision. The last two are builtin now.
    'torch_compile', 'torch_compile_backend', 'torch_compile_mode', 'accelerator_config', 'auto_find_batch_size',
    'ds3_gather_for_generation', 'eval_accumulation_steps', 'bf16_full_eval', 'fp16_full_eval',
    'average_tokens_across_devices', 'use_logits_to_keep')
# Training options retired because the current path already behaves that way or folds them into
# another flag; each is refused with the migration note in REMOVED_OPTION_REPLACEMENTS.
TRAIN_REMOVED_OPTION_FIELDS: Tuple[str, ...] = (
    'prediction_loss_only', 'do_eval', 'eval_use_evalscope', 'label_names', 'max_epochs', 'disable_tqdm',
    'check_model', 'lora_dtype', 'eval_metric',
    # merge_lora is a save/export-time concern (it lives on DeployConfig/ConvertConfig, not on any
    # training Config); a training run merges the adapter implicitly, so refuse it here with the
    # migration note instead of letting it fall through to parse_configs' cryptic "Unrecognized
    # arguments" -- `swift eval` already classifies it, this makes the training commands consistent.
    'merge_lora')
TUNER_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'lisa_activated_layers', 'lisa_step_interval', 'lora_ga_batch_size', 'lora_ga_iters', 'lora_ga_max_length',
    'lora_ga_direction', 'lora_ga_scale', 'lora_ga_stable_gamma', 'fourier_n_frequency', 'fourier_scaling',
    'boft_block_size', 'boft_block_num', 'boft_n_butterfly_factor', 'boft_dropout', 'vera_rank',
    'vera_projection_prng_key', 'vera_dropout', 'vera_d_initial', 'adapter_act', 'adapter_length',
    'llamapro_num_new_blocks', 'llamapro_num_groups', 'reft_layer_key', 'reft_layers', 'reft_rank',
    'reft_intervention_type', 'reft_args')
TRAINING_GENERATION_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'top_k', 'top_p', 'repetition_penalty', 'num_beams', 'stream', 'stop_words', 'logprobs', 'top_logprobs',
    'structured_outputs_regex', 'generation_config',
    # A training run that generates during eval sets the decoding knobs inside --eval_generation_config.
    'temperature', 'max_new_tokens')
MCORE_TRAINING_CHECKPOINT_FIELDS: Tuple[str, ...] = ('mcore_model', 'mcore_adapter')
MCORE_REFERENCE_CHECKPOINT_FIELDS: Tuple[str, ...] = ('mcore_ref_model', 'mcore_ref_adapter')
CHECKPOINT_FIELDS: Tuple[str, ...] = (
    'output_dir', 'save_strategy', 'save_steps', 'save_total_limit', 'safe_serialization', 'max_shard_size',
    'save_on_each_node', 'save_only_model', 'resume_from_checkpoint', 'resume_only_model', 'ignore_data_skip',
    'push_to_hub', 'hub_model_id', 'hub_private_repo', 'hub_strategy', 'hub_revision', 'hub_always_push',
    'no_save_optim', 'no_save_rng', 'no_load_optim', 'no_load_rng', 'save_safetensors',
    'use_persistent_ckpt_worker', 'dist_ckpt_optim_fully_reshardable',
    'distrib_optim_fully_reshardable_mem_efficient', 'dist_ckpt_save_pre_mcore_014', 'add_version',
    'create_checkpoint_symlink', 'use_flash_ckpt', 'load_args', 'load_data_args')
CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'save_on_each_node', 'use_persistent_ckpt_worker', 'dist_ckpt_optim_fully_reshardable',
    'distrib_optim_fully_reshardable_mem_efficient', 'dist_ckpt_save_pre_mcore_014', 'create_checkpoint_symlink',
    'use_flash_ckpt')
CHECKPOINT_ARGS_RESTORE_FIELDS: Tuple[str, ...] = ('load_args', 'load_data_args')
CHECKPOINT_TRANSFORMERS_STATE_FIELDS: Tuple[str, ...] = (
    'no_save_rng', 'no_load_optim', 'no_load_rng', 'save_safetensors')
CHECKPOINT_EXPORT_SUPPORTED_FIELDS = {
    'output_dir', 'safe_serialization', 'max_shard_size', 'push_to_hub', 'hub_model_id', 'hub_private_repo', 'hub_revision'
}
CHECKPOINT_MERGE_SUPPORTED_FIELDS = {'output_dir', 'safe_serialization', 'max_shard_size'}
# `swift merge` only folds a LoRA adapter into base weights; quantization/format conversion is
# export's job, and a command that generates no text consumes no per-flag decoding knobs.
QUANTIZE_OPERATION_FIELDS: Tuple[str, ...] = (
    'quant_method', 'quant_bits', 'quant_batch_size', 'quant_n_samples', 'group_size', 'hqq_axis',
    'bnb_4bit_compute_dtype', 'bnb_4bit_quant_storage', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant')
CONVERT_OPERATION_FIELDS: Tuple[str, ...] = (
    'to_hf', 'to_mcore', 'to_cached_dataset', 'to_peft_format', 'mcore_model', 'mcore_adapter',
    'test_convert_dtype', 'test_convert_precision', 'template_mode', 'thread_count', 'commit_message')
# Flags the `swift merge` command makes redundant: merging is the command itself, and its overwrite
# knob is spelled --replace_if_exists rather than the export-side --exist_ok.
MERGE_IMPLICIT_FIELDS: Tuple[str, ...] = ('merge_lora', 'exist_ok')


@dataclass(frozen=True)
class LegacyFieldContract:
    """One legacy flag's executable compatibility contract."""

    name: str
    kind: Literal['direct', 'alias', 'unsupported']
    owner: Optional[str] = None
    target: Optional[str] = None
    transform: Optional[str] = None
    reason: Optional[str] = None
    replacement: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind == 'unsupported' and (not self.reason or not self.replacement):
            raise ValueError(f'Unsupported legacy field {self.name!r} needs both reason and replacement.')


_CATEGORY_DETAILS = {
    'unsupported_sampler': (
        'this sampler option is not supported',
        'remove this option and choose a supported inference backend and its options'),
    'unsupported_command': (
        'this option is not supported by the current command',
        'remove this option or pass it to a command that supports the corresponding operation'),
    'removed_option': (
        'this option is obsolete and no longer supported',
        'remove this option and use a currently supported option when needed'),
    'unsupported_training': (
        'this training option is not supported',
        'remove this option and choose a supported training or tuning option'),
    'unsupported_checkpoint': (
        'this checkpoint option or artifact format is not supported',
        'remove this option and use a supported HF-format checkpoint or export workflow'),
}

# argument names: from v4 to v5 mapping
LEGACY_ALIASES: Dict[str, str] = {
    'lr': 'learning_rate',
    'train_iters': 'max_steps',
    'micro_batch_size': 'per_device_train_batch_size',
    'lr_warmup_fraction': 'warmup_ratio',
    'lr_warmup_iters': 'warmup_steps',
    'adam_eps': 'adam_epsilon',
    'clip_grad': 'max_grad_norm',
    'attention_backend': 'attn_impl',
    'bf16': 'torch_dtype',
    'fp16': 'torch_dtype',
    'response_length': 'max_completion_length',
    'use_ray': 'mode',
    'sampler_engine': 'sampler',
    'infer_backend': 'sampler',
    # The trainer/sampler GPU-placement field is backend-neutral `sampler_mode` now; v4's --vllm_mode
    # folds onto it here.
    'vllm_mode': 'sampler_mode',
    'tuner_type': 'tuner',
    'reward_funcs': 'orm',
    'reward_weights': 'orm_weights',
    'prm_funcs': 'prm',
    'lr_scheduler_type': 'lr_scheduler',
    # The RLConfig field dropped the "hf" from the v4 flag, so --rlhf_type folds onto `rl_type`.
    'rlhf_type': 'rl_type',
    # Only TrainConfig's loss was renamed; RLConfig.loss_type keeps its name. normalize_argv and
    # build_legacy_contract suppress this alias where loss_type is itself a live field (rlhf).
    'loss_type': 'loss',
    # The v4 Megatron boolean --finetune folds onto the resume contract: true = a weights-only new
    # run, false = a full resume (which then requires --resume_from_checkpoint).
    'finetune': 'resume_only_model',
}

CHECKPOINT_NON_TRAINING_FIELDS: Tuple[str, ...] = tuple(
    name for name in CHECKPOINT_FIELDS if name not in CHECKPOINT_ARGS_RESTORE_FIELDS)
CHECKPOINT_EXPORT_UNSUPPORTED_FIELDS: Tuple[str, ...] = tuple(
    name for name in CHECKPOINT_FIELDS
    if name not in CHECKPOINT_EXPORT_SUPPORTED_FIELDS and name not in CHECKPOINT_ARGS_RESTORE_FIELDS)
CHECKPOINT_MERGE_UNSUPPORTED_FIELDS: Tuple[str, ...] = tuple(
    name for name in CHECKPOINT_FIELDS
    if name not in CHECKPOINT_MERGE_SUPPORTED_FIELDS and name not in CHECKPOINT_ARGS_RESTORE_FIELDS)

CLI_LEGACY_ONLY: Dict[str, Mapping[str, Tuple[str, ...]]] = {
    'pt': {
        'unsupported_command': TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_training': TRAIN_UNSUPPORTED_FIELDS + TUNER_UNSUPPORTED_FIELDS
        + TRAINING_GENERATION_UNSUPPORTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_TRANSFORMERS_STATE_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + TRAIN_REMOVED_OPTION_FIELDS,
    },
    'sft': {
        'unsupported_command': TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_training': TRAIN_UNSUPPORTED_FIELDS + TUNER_UNSUPPORTED_FIELDS
        + TRAINING_GENERATION_UNSUPPORTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_TRANSFORMERS_STATE_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + TRAIN_REMOVED_OPTION_FIELDS,
    },
    'rl': {
        'unsupported_command': TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_training': TRAIN_UNSUPPORTED_FIELDS + TUNER_UNSUPPORTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_TRANSFORMERS_STATE_FIELDS
        + MCORE_REFERENCE_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + TRAIN_REMOVED_OPTION_FIELDS + ('seq_kd',)
        + RLHF_REMOVED_ROLLOUT_FIELDS + RLHF_REMOVED_RL_KNOBS + RLHF_REMOVED_EVAL_FIELDS
        + RLHF_REMOVED_SERVER_FIELDS + REWARD_MODEL_RETIRED_FIELDS,
    },
    'megatron_pt': {
        'unsupported_command': TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + MCORE_TRAINING_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + TRAIN_REMOVED_OPTION_FIELDS,
    },
    'megatron_sft': {
        'unsupported_command': TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + MCORE_TRAINING_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + TRAIN_REMOVED_OPTION_FIELDS,
    },
    'megatron_rl': {
        'unsupported_command': TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + MCORE_TRAINING_CHECKPOINT_FIELDS
        + MCORE_REFERENCE_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + TRAIN_REMOVED_OPTION_FIELDS + ('seq_kd',)
        + RLHF_REMOVED_ROLLOUT_FIELDS + RLHF_REMOVED_RL_KNOBS + RLHF_REMOVED_EVAL_FIELDS
        + RLHF_REMOVED_SERVER_FIELDS + REWARD_MODEL_RETIRED_FIELDS,
    },
    'infer': {
        'unsupported_sampler': UNSUPPORTED_SAMPLER_FIELDS,
        # infer consumes the same two distributed knobs a training run does: its Ray sampling entry names
        # the session (--ray_exp_name -> twinkle.initialize); --ddp_timeout -> the strategy timeout env.
        'unsupported_command': TRAIN_UNSUPPORTED_DISTRIBUTED_FIELDS + CHECKPOINT_NON_TRAINING_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + ('data_range',),
    },
    'deploy': {
        'unsupported_sampler': UNSUPPORTED_SAMPLER_FIELDS,
        'unsupported_command': SERVING_DISTRIBUTED_FIELDS + CHECKPOINT_NON_TRAINING_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'eval': {
        'unsupported_sampler': UNSUPPORTED_SAMPLER_FIELDS,
        'unsupported_command': SERVING_DISTRIBUTED_FIELDS + CHECKPOINT_NON_TRAINING_FIELDS + EVAL_SERVING_FIELDS
        + ('merge_lora',),
        'removed_option': REMOVED_OPTION_FIELDS + EVAL_REMOTE_EVAL_FIELDS + GENERATION_OPTION_FIELDS,
    },
    'export': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS + CHECKPOINT_EXPORT_UNSUPPORTED_FIELDS
        + GENERATION_OPTION_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'merge': {
        'unsupported_command': SERVING_DISTRIBUTED_FIELDS + CHECKPOINT_MERGE_UNSUPPORTED_FIELDS
        + QUANTIZE_OPERATION_FIELDS + CONVERT_OPERATION_FIELDS + GENERATION_OPTION_FIELDS + MERGE_IMPLICIT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
}


def classified_fields(command: str) -> set:
    """Return the disjoint legacy-only field set accounted for by ``command``."""
    groups = CLI_LEGACY_ONLY.get(command, {})
    result = set()
    for fields in groups.values():
        overlap = result.intersection(fields)
        if overlap:
            raise AssertionError(f'{command} legacy classifications overlap: {sorted(overlap)}')
        result.update(fields)
    return result


def unsupported_contracts(command: str) -> Dict[str, LegacyFieldContract]:
    """Return detailed contracts for flags deliberately unavailable on ``command``."""
    result = {}
    for category, names in CLI_LEGACY_ONLY.get(command, {}).items():
        reason, replacement = _CATEGORY_DETAILS[category]
        for name in names:
            if name in result:
                raise AssertionError(f'{command} legacy field {name!r} is classified more than once.')
            result[name] = LegacyFieldContract(
                name=name,
                kind='unsupported',
                reason=reason,
                replacement=REMOVED_OPTION_REPLACEMENTS.get(name, replacement),
            )
    return result


def build_legacy_contract(command: str,
                          legacy_fields: Iterable[str],
                          config_classes: Sequence[Type],
                          *,
                          field_owners: Optional[Mapping[str, Type]] = None) -> Tuple[Dict[str, LegacyFieldContract], set]:
    """Classify legacy fields as direct, aliased, unsupported, or unaccounted."""
    owners = dict(field_owners or {})
    config_owners: Dict[str, Type] = {}
    for config_class in config_classes:
        for config_field in dataclasses.fields(config_class):
            config_owners.setdefault(config_field.name, config_class)
            if owners.get(config_field.name) is config_class:
                config_owners[config_field.name] = config_class

    legacy_set = set(legacy_fields)
    contracts = {name: item for name, item in unsupported_contracts(command).items() if name in legacy_set}
    for name in legacy_set:
        if name in contracts:
            continue
        target = LEGACY_ALIASES.get(name)
        # A legacy spelling that is itself a live field of a DIFFERENT Config than the alias target
        # is that Config's own flag, not a rename; classify it direct to match normalize_argv.
        cross_owner_field = config_owners.get(name) is not None and config_owners.get(target) is not config_owners[name]
        if target in config_owners and not cross_owner_field:
            owner = config_owners[target]
            if name in {'bf16', 'fp16'}:
                transform = 'precision_bool_to_dtype'
            elif name == 'use_ray':
                transform = 'boolean_to_mode'
            else:
                transform = 'identity'
            contracts[name] = LegacyFieldContract(
                name=name,
                kind='alias',
                owner=owner.__name__,
                target=target,
                transform=transform,
            )
            continue
        owner = config_owners.get(name)
        if owner is not None:
            contracts[name] = LegacyFieldContract(
                name=name,
                kind='direct',
                owner=owner.__name__,
                target=name,
            )
    return contracts, legacy_set - set(contracts)


def reject_legacy_only_flags(command: str, argv: Sequence[str]) -> None:
    """Reject known unsupported options with a reason and an alternative."""
    from .parser import flag_names

    passed = flag_names(argv)
    for name, contract in unsupported_contracts(command).items():
        if name not in passed:
            continue
        raise ValueError(
            f'--{name} is unsupported by `{command}`: {contract.reason}. Alternative: {contract.replacement}.')


def flatten(groups: Mapping[str, Iterable[str]]) -> set:
    """Flatten a classification mapping for audit tests."""
    return {name for names in groups.values() for name in names}
