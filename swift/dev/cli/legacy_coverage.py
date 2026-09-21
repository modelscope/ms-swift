"""Executable accounting for legacy CLI fields that are not dev Config inputs."""
from __future__ import annotations
from typing import Dict, Iterable, Mapping, Sequence, Tuple

LMDEPLOY_FIELDS: Tuple[str, ...] = (
    'lmdeploy_cache_max_entry_count', 'lmdeploy_quant_policy', 'lmdeploy_session_len', 'lmdeploy_tp',
    'lmdeploy_vision_batch_size')
LOAD_QUANTIZATION_FIELDS: Tuple[str, ...] = (
    'bnb_4bit_compute_dtype', 'bnb_4bit_quant_storage', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant', 'hqq_axis',
    'quant_bits', 'quant_method')
COMMON_SUPERSEDED_FIELDS: Tuple[str, ...] = ('ignore_args_error', 'use_swift_lora')
SERVING_DISTRIBUTED_FIELDS: Tuple[str, ...] = (
    'ddp_backend', 'ddp_timeout', 'device_groups', 'ray_exp_name', 'use_ray')
ROLLOUT_INHERITED_INFER_FIELDS: Tuple[str, ...] = (
    'infer_backend', 'max_batch_size', 'merge_lora', 'metric', 'reranker_use_activation', 'result_path',
    'val_dataset_sample', 'write_batch_size')

CLI_LEGACY_ONLY: Dict[str, Mapping[str, Tuple[str, ...]]] = {
    'infer': {
        'builder_contract': LOAD_QUANTIZATION_FIELDS,
        'excluded_lmdeploy': LMDEPLOY_FIELDS,
        'superseded': COMMON_SUPERSEDED_FIELDS,
    },
    'deploy': {
        'builder_contract': LOAD_QUANTIZATION_FIELDS,
        'excluded_lmdeploy': LMDEPLOY_FIELDS,
        'inherited_non_applicable': SERVING_DISTRIBUTED_FIELDS,
        'superseded': COMMON_SUPERSEDED_FIELDS,
    },
    'rollout': {
        'builder_contract': LOAD_QUANTIZATION_FIELDS,
        'excluded_lmdeploy': LMDEPLOY_FIELDS,
        'inherited_non_applicable': SERVING_DISTRIBUTED_FIELDS + ROLLOUT_INHERITED_INFER_FIELDS,
        'superseded': COMMON_SUPERSEDED_FIELDS,
    },
    'sample': {
        'builder_contract': LOAD_QUANTIZATION_FIELDS,
        'superseded': COMMON_SUPERSEDED_FIELDS,
    },
    'eval': {
        'builder_contract': LOAD_QUANTIZATION_FIELDS,
        'excluded_lmdeploy': LMDEPLOY_FIELDS,
        'inherited_non_applicable': SERVING_DISTRIBUTED_FIELDS,
        'superseded': COMMON_SUPERSEDED_FIELDS,
    },
    'app': {
        'builder_contract': LOAD_QUANTIZATION_FIELDS,
        'excluded_lmdeploy': LMDEPLOY_FIELDS,
        'inherited_non_applicable': SERVING_DISTRIBUTED_FIELDS,
        'superseded': COMMON_SUPERSEDED_FIELDS,
    },
    'export': {'superseded': COMMON_SUPERSEDED_FIELDS},
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


def reject_legacy_only_flags(command: str, argv: Sequence[str]) -> None:
    """Reject known gaps before argparse can report an opaque unknown-option error."""
    from swift.dev.cli.parser import flag_names

    passed = flag_names(argv)
    for category, names in CLI_LEGACY_ONLY.get(command, {}).items():
        matched = sorted(passed.intersection(names))
        if not matched:
            continue
        if category == 'excluded_lmdeploy':
            raise ValueError(f'{matched} belong to lmdeploy, which is intentionally excluded from the v5 CLI.')
        if category == 'builder_contract':
            raise NotImplementedError(
                f'{matched} require load-time quantization, but the current dev model-builder contract does not '
                'accept QuantizeConfig. The flags are refused instead of being silently ignored.')
        if category == 'inherited_non_applicable':
            raise ValueError(f'{matched} are inherited legacy options that do not apply to standalone `{command}`.')
        raise ValueError(f'{matched} are superseded legacy options and are not accepted by the strict v5 CLI.')


def flatten(groups: Mapping[str, Iterable[str]]) -> set:
    """Flatten a classification mapping for audit tests."""
    return {name for names in groups.values() for name in names}
