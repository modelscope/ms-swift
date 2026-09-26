"""Self-parse of argv straight into dev's atomic Configs.

The dev CLI used to lean on legacy ``SftArguments``/``ExportArguments``: parse the legacy surface,
then copy same-named fields onto the Configs. That bridge dragged in ``swift/arguments`` and left the
legacy ``__post_init__`` side effects (output_dir versioning, plugin import, hub login, ...) behind.
This module removes the bridge: ``HfArgumentParser`` builds argparse options directly from the Config
dataclasses, so the Config *is* the argument surface and its defaults are the single source of truth.

Container annotations that ``HfArgumentParser`` cannot introspect are rewritten only for argparse;
the parsed values are then restored from each Config's original runtime annotation.
"""
from __future__ import annotations
import os
import sys
from contextlib import contextmanager
from dataclasses import MISSING, field, fields, make_dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, get_args, get_origin, get_type_hints

import json
from transformers import HfArgumentParser

from .legacy_coverage import LEGACY_ALIASES


def _mapping_hint(annotation):
    """Return the mapping member and whether the annotation also permits a string."""
    if annotation is dict or get_origin(annotation) is dict:
        return annotation, False
    args = get_args(annotation)
    mapping = next((arg for arg in args if arg is dict or get_origin(arg) is dict), None)
    return mapping, str in args


@contextmanager
def _patch_type_hints():
    """Present unsupported container annotations to ``HfArgumentParser`` as CLI-friendly types."""
    from transformers import hf_argparser
    origin_get_type_hints = hf_argparser.get_type_hints
    str_list_none = Union[str, List[str], type(None)]

    def get_type_hints(*args, **kwargs):
        hints = origin_get_type_hints(*args, **kwargs)
        for k, v in hints.items():
            union_args = get_args(v)
            mapping, _ = _mapping_hint(v)
            list_args = [arg for arg in union_args if get_origin(arg) is list]
            scalar_args = [arg for arg in union_args if arg is not type(None) and get_origin(arg) is not list]
            if mapping is not None:
                hints[k] = Optional[str] if type(None) in union_args else str
            elif v == str_list_none:
                hints[k] = Optional[str]
            elif len(list_args) == 1 and len(scalar_args) == 1:
                # HfArgumentParser rejects Union[scalar, List[scalar]]. Parse the CLI as a list so
                # repeated values remain expressible; an untouched scalar dataclass default is kept.
                hints[k] = list_args[0]
        return hints

    hf_argparser.get_type_hints = get_type_hints
    try:
        yield
    finally:
        hf_argparser.get_type_hints = origin_get_type_hints


def _coerce_mapping_item(value, annotation):
    """Restore key/value scalar types lost when a JSON object crosses the CLI boundary."""
    if annotation in (Any, object) or value is None:
        return value
    origin = get_origin(annotation)
    if origin is Union:
        for member in get_args(annotation):
            if member is type(None):
                continue
            try:
                return _coerce_mapping_item(value, member)
            except (TypeError, ValueError):
                continue
        raise TypeError(f'{value!r} does not match {annotation}.')
    if annotation is dict or origin is dict:
        return _coerce_mapping(value, annotation)
    if origin is list:
        (item_type, ) = get_args(annotation) or (Any, )
        return [_coerce_mapping_item(item, item_type) for item in value]
    if annotation in (str, int, float):
        return value if isinstance(value, annotation) else annotation(value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        parsed = _boolean_value(value) if isinstance(value, str) else None
        if parsed is None:
            raise ValueError(f'{value!r} is not a boolean.')
        return parsed
    return value


def _coerce_mapping(value, annotation) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f'expected a JSON object, got {type(value).__name__}')
    args = get_args(annotation)
    if len(args) != 2:
        return value
    key_type, value_type = args
    return {
        _coerce_mapping_item(key, key_type): _coerce_mapping_item(item, value_type)
        for key, item in value.items()
    }


def _restore_config_types(config) -> None:
    """Restore CLI strings to the runtime types declared by one Config dataclass."""
    from swift.dev.utils import json_parse_to_dict

    hints = get_type_hints(type(config))
    for config_field in fields(config):
        annotation = hints.get(config_field.name, config_field.type)
        mapping, allow_string = _mapping_hint(annotation)
        if mapping is None:
            continue
        value = getattr(config, config_field.name)
        if value is None:
            continue
        original = value
        try:
            if isinstance(value, str):
                value = json_parse_to_dict(value, strict=not allow_string)
            if not isinstance(value, dict):
                if allow_string and isinstance(value, str):
                    continue
                raise TypeError(f'expected a JSON object or JSON file, got {original!r}')
            setattr(config, config_field.name, _coerce_mapping(value, mapping))
        except (TypeError, ValueError) as error:
            raise TypeError(f'Invalid --{config_field.name}: {error}') from error


def resolve_argv(argv: Optional[List[str]] = None) -> List[str]:
    """Resolve explicit arguments or fall back to the process command line."""
    return list(sys.argv[1:] if argv is None else argv)


def flag_names(argv: Sequence[str]) -> set:
    """Return normalized ``--flag`` names present in argv."""
    return {token[2:].split('=', 1)[0].replace('-', '_') for token in argv if token.startswith('--')}


def select_tuner(tuner):
    """Represent full-parameter runs without a TunerConfig."""
    return None if tuner.tuner_type == 'full' else tuner


# Public compatibility export; the declaration itself lives with the executable legacy contract.
CLI_ALIASES = {name: target for name, target in LEGACY_ALIASES.items() if name not in {'bf16', 'fp16', 'response_length'}}
_PRECISION_ALIASES = {'bf16': 'bfloat16', 'fp16': 'float16'}
_TRUE_VALUES = {'1', 'true', 'True', 'TRUE', 'yes', 'y', 'on'}
_FALSE_VALUES = {'0', 'false', 'False', 'FALSE', 'no', 'n', 'off'}


def _boolean_value(value: str):
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return None


def _comparable_value(value: str):
    # Keep numeric 0/1 numeric here: an alias pair such as --lr 1 --learning-rate 1.0 must compare
    # equal. Boolean-only aliases call _boolean_value explicitly before reaching this helper.
    lowered = value.lower()
    if lowered in _TRUE_VALUES - {'1'}:
        return True
    if lowered in _FALSE_VALUES - {'0'}:
        return False
    try:
        return Decimal(value)
    except InvalidOperation:
        return value


def normalize_argv(argv: Sequence[str], available_fields: Sequence[str]) -> List[str]:
    """Rewrite legacy aliases to canonical Config flags and reject conflicting double writes.

    Scalar aliases accept both ``--name value`` and ``--name=value``. Hyphens and underscores are
    equivalent. If an alias and its canonical spelling are both explicit, equal values are accepted
    while different values fail before argparse can silently let the last occurrence win.
    """
    available = set(available_fields)
    aliases = {alias: target for alias, target in CLI_ALIASES.items() if target in available}
    if 'torch_dtype' in available:
        aliases.update({name: 'torch_dtype' for name in _PRECISION_ALIASES})

    result: List[str] = []
    seen: Dict[str, Tuple[str, object]] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith('--'):
            result.append(token)
            index += 1
            continue

        option, separator, inline_value = token[2:].partition('=')
        source = option.replace('-', '_')
        target = aliases.get(source, source)
        has_value = bool(separator)
        value = inline_value if has_value else None
        if not has_value and index + 1 < len(argv) and not argv[index + 1].startswith('--'):
            value = argv[index + 1]
            has_value = True
            index += 1

        if source in _PRECISION_ALIASES and target == 'torch_dtype':
            enabled = True if value is None else _boolean_value(value)
            if enabled is None:
                raise ValueError(f'--{option} expects a boolean value, got {value!r}.')
            if not enabled:
                index += 1
                continue
            value = _PRECISION_ALIASES[source]
            has_value = True
        elif source == 'use_ray' and target == 'mode':
            enabled = True if value is None else _boolean_value(value)
            if enabled is None:
                raise ValueError(f'--{option} expects a boolean value, got {value!r}.')
            value = 'ray' if enabled else 'local'
            has_value = True

        if target in available and (source in aliases or source == target):
            if not has_value:
                comparable = True
            else:
                comparable = _comparable_value(value)
            previous = seen.get(target)
            if previous is not None and previous[1] != comparable:
                raise ValueError(
                    f'Conflicting values for --{target}: --{previous[0]}={previous[1]!r} and '
                    f'--{option}={comparable!r}. Use one spelling or pass the same value.')
            seen[target] = (option, comparable)

        result.append(f'--{target}')
        if has_value:
            result.append(value)
        index += 1
    return result


def _merged_parse_class(config_classes: Sequence[Type], field_owners: Optional[Dict[str, Type]] = None):
    """Build one argparse dataclass from several Configs, resolving duplicate field names explicitly."""
    owners = dict(field_owners or {})
    specs = []
    seen = {}
    for config_class in config_classes:
        hints = get_type_hints(config_class)
        for config_field in fields(config_class):
            name = config_field.name
            if name in seen:
                preferred = owners.get(name)
                if preferred is not None and preferred is config_class:
                    specs[seen[name]] = None
                else:
                    continue
            kwargs = {'metadata': dict(config_field.metadata)}
            if config_field.default is not MISSING:
                kwargs['default'] = config_field.default
            elif config_field.default_factory is not MISSING:
                kwargs['default_factory'] = config_field.default_factory
            spec = (name, hints.get(name, config_field.type), field(**kwargs))
            if name in seen:
                specs.append(spec)
                seen[name] = len(specs) - 1
            else:
                seen[name] = len(specs)
                specs.append(spec)
    return make_dataclass('MergedCliConfig', [spec for spec in specs if spec is not None])


_CHECKPOINT_FORCE_LOAD_FIELDS = {
    'model_config': ('task_type',),
    'quantize_config': ('bnb_4bit_quant_type', 'bnb_4bit_use_double_quant'),
    'tuner_config': ('tuner_type',),
}
_CHECKPOINT_DATA_FIELDS = (
    'dataset', 'val_dataset', 'cached_dataset', 'cached_val_dataset', 'split_dataset_ratio', 'data_seed',
    'dataset_num_proc', 'load_from_cache_file', 'dataset_shuffle', 'val_dataset_shuffle', 'streaming', 'interleave_prob',
    'stopping_strategy', 'shuffle_buffer_size', 'download_mode', 'columns', 'strict', 'remove_unused_columns',
    'disable_auto_column_mapping', 'model_name', 'model_author', 'custom_dataset_info')
_CHECKPOINT_LOAD_FIELDS = {
    'model_config': (
        'model', 'model_type', 'model_revision', 'torch_dtype', 'attn_impl', 'experts_impl',
        'new_special_tokens', 'num_labels', 'problem_type', 'rope_scaling', 'max_model_len'),
    'plugin_config': ('external_plugins', ),
    'quantize_config': (
        'quant_method', 'quant_bits', 'hqq_axis', 'bnb_4bit_compute_dtype', 'bnb_4bit_quant_storage'),
    'template_config': (
        'template', 'system', 'truncation_strategy', 'agent_template', 'norm_bbox', 'use_chat_template',
        'response_prefix'),
}


def _config_mapping(configs: Sequence[Any]) -> Dict[str, Any]:
    """Index parsed Config instances by their conventional CLI mapping names."""
    result = {}
    for config in configs:
        name = type(config).__name__
        if name.endswith('Config'):
            result[f'{name[:-6].lower()}_config'] = config
    return result


def _resolve_checkpoint_args_source(configs: Dict[str, Any]) -> Optional[str]:
    """Resolve the first model/adapter checkpoint that contains ``args.json``."""
    checkpoint = configs['checkpoint_config']
    model = configs.get('model_config')
    tuner = configs.get('tuner_config')
    convert = configs.get('convert_config')
    candidates = []
    if tuner is not None:
        for index, adapter in enumerate(tuner.adapters or []):
            candidates.append((tuner.adapters, index, adapter, True, None))
    if convert is not None and convert.mcore_adapter:
        candidates.append((convert, 'mcore_adapter', convert.mcore_adapter, False, None))
    if checkpoint.resume_from_checkpoint:
        candidates.append((checkpoint, 'resume_from_checkpoint', checkpoint.resume_from_checkpoint, False, None))
    if model is not None and model.model:
        candidates.append((model, 'model', model.model, True, model.model_revision))
    if convert is not None and convert.mcore_model:
        candidates.append((convert, 'mcore_model', convert.mcore_model, False, None))

    dataset = configs.get('dataset_config')
    remote_only = bool(getattr(configs.get('eval_config'), 'eval_url', None))
    for owner, key, source, downloadable, revision in candidates:
        resolved = source
        args_path = os.path.join(resolved, 'args.json')
        if not os.path.isfile(args_path) and downloadable and not remote_only:
            from swift.dev.utils.hub import safe_snapshot_download
            resolved = safe_snapshot_download(
                source,
                revision=revision,
                use_hf=getattr(dataset, 'use_hf', None),
                hub_token=getattr(dataset, 'hub_token', None))
            if isinstance(owner, list):
                owner[key] = resolved
            else:
                setattr(owner, key, resolved)
            args_path = os.path.join(resolved, 'args.json')
        if os.path.isfile(args_path):
            return args_path
    return None


def _restore_checkpoint_args(configs: Sequence[Any], *, default_load_args: bool) -> None:
    """Apply legacy ``load_args_from_ckpt`` semantics to atomic dev Configs."""
    mapped = _config_mapping(configs)
    checkpoint = mapped.get('checkpoint_config')
    if checkpoint is None:
        return
    explicit = getattr(checkpoint, '_explicit_fields', set())
    if 'load_args' not in explicit:
        checkpoint.load_args = default_load_args
    if not checkpoint.load_args:
        return
    args_path = _resolve_checkpoint_args_source(mapped)
    if args_path is None:
        return
    with open(args_path, encoding='utf-8') as file:
        old_args = json.load(file)
    if not isinstance(old_args, dict):
        raise TypeError(f'Checkpoint arguments must be a JSON object: {args_path}')

    load_fields = {name: list(names) for name, names in _CHECKPOINT_LOAD_FIELDS.items()}
    swift_version = old_args.get('swift_version')
    if swift_version is None:
        load_fields['model_config'].remove('model_type')
    else:
        from packaging import version
        if version.parse(swift_version) < version.parse('4.0.0.dev'):
            load_fields['model_config'].remove('model_type')
    for config_name, names in _CHECKPOINT_FORCE_LOAD_FIELDS.items():
        config = mapped.get(config_name)
        if config is None:
            continue
        for name in names:
            if old_args.get(name) is not None:
                setattr(config, name, old_args[name])
    for config_name, names in load_fields.items():
        config = mapped.get(config_name)
        if config is None:
            continue
        for name in names:
            old_value = old_args.get(name)
            value = getattr(config, name, None)
            if old_value is not None and (value is None or isinstance(value, (list, tuple)) and not value):
                setattr(config, name, old_value)
    if checkpoint.load_data_args:
        dataset = mapped.get('dataset_config')
        if dataset is not None:
            for name in _CHECKPOINT_DATA_FIELDS:
                if old_args.get(name) is not None:
                    setattr(dataset, name, old_args[name])


def parse_configs(
    config_classes: Sequence[Type],
    argv: Optional[List[str]] = None,
    *,
    field_owners: Optional[Dict[str, Type]] = None,
    load_args_default: bool = False,
) -> Tuple[list, List[str]]:
    """Parse argv into Config instances, including Config sets with duplicate field names.

    ``field_owners`` selects which Config owns a duplicate CLI spelling. Other Configs retain their
    own defaults; command-specific post-processing may deliberately mirror the parsed value when one
    flag controls more than one runtime component.
    """
    effective_argv = resolve_argv(argv)
    merged_class = _merged_parse_class(config_classes, field_owners)
    effective_argv = normalize_argv(effective_argv, [config_field.name for config_field in fields(merged_class)])
    explicit_fields = flag_names(effective_argv)
    with _patch_type_hints():
        parser = HfArgumentParser(merged_class)
    merged, remaining = parser.parse_args_into_dataclasses(effective_argv, return_remaining_strings=True)
    values = vars(merged)
    owners = dict(field_owners or {})
    first_owner = {}
    for config_class in config_classes:
        for config_field in fields(config_class):
            first_owner.setdefault(config_field.name, config_class)

    configs = []
    for config_class in config_classes:
        kwargs = {}
        for config_field in fields(config_class):
            owner = owners.get(config_field.name, first_owner[config_field.name])
            if owner is config_class and config_field.name in values:
                kwargs[config_field.name] = values[config_field.name]
        config = config_class(**kwargs)
        config._explicit_fields = {config_field.name for config_field in fields(config_class)
                                   if config_field.name in explicit_fields}
        configs.append(config)
    if not remaining:
        _restore_checkpoint_args(configs, default_load_args=load_args_default)
    for config in configs:
        _restore_config_types(config)
    return configs, remaining


def parse_configs_strict(
    config_classes: Sequence[Type],
    argv: Optional[List[str]] = None,
    *,
    command: str,
    field_owners: Optional[Dict[str, Type]] = None,
    load_args_default: bool = False,
) -> list:
    """Parse Configs and reject every unrecognized argument."""
    configs, remaining = parse_configs(
        config_classes, argv, field_owners=field_owners, load_args_default=load_args_default)
    if remaining:
        raise ValueError(f'Unrecognized arguments for {command}: {remaining}.')
    return configs
