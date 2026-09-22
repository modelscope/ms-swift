"""Compatibility accounting for legacy CLI fields."""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Type

UNSUPPORTED_SAMPLER_FIELDS: Tuple[str, ...] = (
    'lmdeploy_cache_max_entry_count', 'lmdeploy_quant_policy', 'lmdeploy_session_len', 'lmdeploy_tp',
    'lmdeploy_vision_batch_size')
REMOVED_OPTION_FIELDS: Tuple[str, ...] = ('ignore_args_error', 'use_swift_lora')
UNSUPPORTED_DISTRIBUTED_FIELDS: Tuple[str, ...] = ('ddp_backend', 'ddp_timeout', 'device_groups', 'ray_exp_name')
SERVING_DISTRIBUTED_FIELDS: Tuple[str, ...] = UNSUPPORTED_DISTRIBUTED_FIELDS + ('use_ray',)
TRAIN_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'optim_target_modules', 'log_on_each_node', 'include_num_input_tokens_seen', 'log_level', 'log_level_replica',
    'project', 'trackio_space_id', 'trackio_bucket_id', 'trackio_static_space_id', 'eval_do_concat_batches',
    'eval_use_gather_object', 'include_for_metrics', 'batch_eval_metrics', 'enable_jit_checkpoint',
    'restore_callback_states_from_checkpoint', 'use_cpu', 'parallelism_config', 'train_sampling_strategy',
    'length_column_name', 'debug', 'skip_memory_metrics', 'do_train', 'do_predict', 'sortish_sampler')
TUNER_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'lisa_activated_layers', 'lisa_step_interval', 'lora_ga_batch_size', 'lora_ga_iters', 'lora_ga_max_length',
    'lora_ga_direction', 'lora_ga_scale', 'lora_ga_stable_gamma', 'fourier_n_frequency', 'fourier_scaling',
    'boft_block_size', 'boft_block_num', 'boft_n_butterfly_factor', 'boft_dropout', 'vera_rank',
    'vera_projection_prng_key', 'vera_dropout', 'vera_d_initial', 'adapter_act', 'adapter_length',
    'llamapro_num_new_blocks', 'llamapro_num_groups', 'reft_layer_key', 'reft_layers', 'reft_rank',
    'reft_intervention_type', 'reft_args')
TRAINING_GENERATION_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'top_k', 'top_p', 'repetition_penalty', 'num_beams', 'stream', 'stop_words', 'logprobs', 'top_logprobs',
    'structured_outputs_regex', 'generation_config', 'generation_max_length', 'generation_num_beams')
MCORE_TRAINING_CHECKPOINT_FIELDS: Tuple[str, ...] = ('mcore_model', 'mcore_adapter')
MCORE_REFERENCE_CHECKPOINT_FIELDS: Tuple[str, ...] = ('mcore_ref_model', 'mcore_ref_adapter')
CHECKPOINT_FIELDS: Tuple[str, ...] = (
    'output_dir', 'save_strategy', 'save_steps', 'save_total_limit', 'safe_serialization', 'max_shard_size',
    'save_on_each_node', 'save_only_model', 'resume_from_checkpoint', 'resume_only_model', 'ignore_data_skip',
    'push_to_hub', 'hub_model_id', 'hub_private_repo', 'hub_strategy', 'hub_revision', 'hub_always_push',
    'no_save_optim', 'no_save_rng', 'no_load_optim', 'no_load_rng', 'async_save', 'save_safetensors',
    'use_persistent_ckpt_worker', 'dist_ckpt_optim_fully_reshardable',
    'distrib_optim_fully_reshardable_mem_efficient', 'dist_ckpt_save_pre_mcore_014', 'add_version',
    'create_checkpoint_symlink', 'use_flash_ckpt', 'load_args', 'load_data_args')
CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
    'save_on_each_node', 'async_save', 'use_persistent_ckpt_worker', 'dist_ckpt_optim_fully_reshardable',
    'distrib_optim_fully_reshardable_mem_efficient', 'dist_ckpt_save_pre_mcore_014', 'create_checkpoint_symlink',
    'use_flash_ckpt')
CHECKPOINT_ARGS_RESTORE_FIELDS: Tuple[str, ...] = ('load_args', 'load_data_args')
CHECKPOINT_HUB_FIELDS: Tuple[str, ...] = (
    'push_to_hub', 'hub_model_id', 'hub_private_repo', 'hub_strategy', 'hub_revision', 'hub_always_push')
CHECKPOINT_TRANSFORMERS_STATE_FIELDS: Tuple[str, ...] = (
    'no_save_rng', 'no_load_optim', 'no_load_rng', 'save_safetensors')
CHECKPOINT_EXPORT_SUPPORTED_FIELDS = {
    'output_dir', 'safe_serialization', 'max_shard_size', 'push_to_hub', 'hub_model_id', 'hub_private_repo', 'hub_revision'
}
CHECKPOINT_MERGE_SUPPORTED_FIELDS = {'output_dir', 'safe_serialization', 'max_shard_size'}


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
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_training': TRAIN_UNSUPPORTED_FIELDS + TUNER_UNSUPPORTED_FIELDS
        + TRAINING_GENERATION_UNSUPPORTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_TRANSFORMERS_STATE_FIELDS
        + CHECKPOINT_HUB_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'sft': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_training': TRAIN_UNSUPPORTED_FIELDS + TUNER_UNSUPPORTED_FIELDS
        + TRAINING_GENERATION_UNSUPPORTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_TRANSFORMERS_STATE_FIELDS
        + CHECKPOINT_HUB_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'rlhf': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_training': TRAIN_UNSUPPORTED_FIELDS + TUNER_UNSUPPORTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_TRANSFORMERS_STATE_FIELDS
        + CHECKPOINT_HUB_FIELDS + MCORE_REFERENCE_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + ('seq_kd',),
    },
    'megatron_pt': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_HUB_FIELDS
        + MCORE_TRAINING_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'megatron_sft': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_HUB_FIELDS
        + MCORE_TRAINING_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'megatron_rlhf': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS,
        'unsupported_checkpoint': CHECKPOINT_RUNTIME_UNSUPPORTED_FIELDS + CHECKPOINT_HUB_FIELDS
        + MCORE_TRAINING_CHECKPOINT_FIELDS + MCORE_REFERENCE_CHECKPOINT_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + ('seq_kd',),
    },
    'infer': {
        'unsupported_sampler': UNSUPPORTED_SAMPLER_FIELDS,
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS + CHECKPOINT_NON_TRAINING_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'deploy': {
        'unsupported_sampler': UNSUPPORTED_SAMPLER_FIELDS,
        'unsupported_command': SERVING_DISTRIBUTED_FIELDS + CHECKPOINT_NON_TRAINING_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'sample': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS + CHECKPOINT_NON_TRAINING_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS + ('data_range',),
    },
    'eval': {
        'unsupported_sampler': UNSUPPORTED_SAMPLER_FIELDS,
        'unsupported_command': SERVING_DISTRIBUTED_FIELDS + CHECKPOINT_NON_TRAINING_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'export': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS + CHECKPOINT_EXPORT_UNSUPPORTED_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'megatron_export': {
        'unsupported_command': UNSUPPORTED_DISTRIBUTED_FIELDS + CHECKPOINT_EXPORT_UNSUPPORTED_FIELDS,
        'removed_option': REMOVED_OPTION_FIELDS,
    },
    'merge_lora': {
        'unsupported_command': SERVING_DISTRIBUTED_FIELDS + CHECKPOINT_MERGE_UNSUPPORTED_FIELDS,
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
                replacement=replacement,
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
        if target in config_owners:
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
