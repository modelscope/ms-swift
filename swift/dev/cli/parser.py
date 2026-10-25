# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import MISSING, field, fields, make_dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, get_args, get_origin, get_type_hints

from transformers import HfArgumentParser

from swift.dev.utils.logger import get_logger

from .legacy_coverage import LEGACY_ALIASES
from swift.dev.config.checkpoint_config import (
    CHECKPOINT_DATA_FIELDS as _CHECKPOINT_DATA_FIELDS,
    CHECKPOINT_FORCE_LOAD_FIELDS as _CHECKPOINT_FORCE_LOAD_FIELDS,
    CHECKPOINT_LOAD_FIELDS as _CHECKPOINT_LOAD_FIELDS,
)

logger = get_logger()


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
            # Any crosses the CLI only as a string spec (name | id | path | class name) the resolve
            # layer parses, so argparse collects strings; the dataclass field keeps Any.
            if v is Any:
                hints[k] = str
                continue
            if get_origin(v) is list and get_args(v) == (Any, ):
                hints[k] = List[str]
                continue
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


def _restore_scalar_or_list(config, name, annotation) -> None:
    """Collapse a lone CLI value back to a scalar for a ``Union[scalar, List[scalar]]`` field.

    The parser view is ``List[scalar]`` so a repeated flag stays expressible, but consumers branch
    on scalar-vs-list (a Megatron router wants a scalar ``moe_aux_loss_coeff``), so unwrap lone values.
    """
    if get_origin(annotation) is not Union:
        return
    union_args = get_args(annotation)
    list_args = [arg for arg in union_args if get_origin(arg) is list]
    scalar_args = [arg for arg in union_args if arg is not type(None) and get_origin(arg) is not list]
    if len(list_args) != 1 or len(scalar_args) != 1:
        return
    value = getattr(config, name)
    if isinstance(value, list) and len(value) == 1:
        setattr(config, name, value[0])


def _restore_config_types(config) -> None:
    """Restore CLI strings to the runtime types declared by one Config dataclass."""
    from swift.dev.utils import json_parse_to_dict

    hints = get_type_hints(type(config))
    for config_field in fields(config):
        annotation = hints.get(config_field.name, config_field.type)
        mapping, allow_string = _mapping_hint(annotation)
        if mapping is None:
            _restore_scalar_or_list(config, config_field.name, annotation)
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
    """Map ``tuner='full'`` to None so downstream reads a missing TunerConfig as no-adapter; GaLore
    is a full-param optimizer technique whose knobs live on TrainConfig, so no config is kept."""
    if tuner.tuner == 'full':
        return None
    return tuner


def normalize_sampler(infer_config) -> None:
    """Fold the legacy ``pt`` sampler value onto the backend name ``transformers`` in place.

    ``run_deploy``'s backend Literal and the engine-arg builders only know the backend names, so
    every command that parses an InferConfig folds the v4 spelling before use (infer/deploy/eval).
    """
    if infer_config.sampler == 'pt':
        infer_config.sampler = 'transformers'


# Boolean legacy flags select an enum value instead of passing one through: bf16/fp16 fold onto
# torch_dtype (off drops the token, keeping the default dtype), use_ray onto mode (off = local).
_BOOLEAN_ALIASES = {
    'bf16': ('torch_dtype', 'bfloat16', None),
    'fp16': ('torch_dtype', 'float16', None),
    'use_ray': ('mode', 'ray', 'local'),
}
# Public compatibility export; the declaration itself lives with the executable legacy contract.
CLI_ALIASES = {name: target for name, target in LEGACY_ALIASES.items() if name not in _BOOLEAN_ALIASES}
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


def _scan_flag(argv: Sequence[str], index: int) -> Tuple[str, Optional[str], bool, int]:
    """Split ``argv[index]`` (plus a following value token) into option/value/has_value/next_index."""
    token = argv[index]
    option, separator, inline_value = token[2:].partition('=')
    has_value = bool(separator)
    value = inline_value if has_value else None
    next_index = index + 1
    if not has_value and next_index < len(argv) and not argv[next_index].startswith('--'):
        value = argv[next_index]
        has_value = True
        next_index += 1
    return option, value, has_value, next_index


def normalize_argv(argv: Sequence[str],
                   available_fields: Sequence[str],
                   field_owners: Optional[Dict[str, Type]] = None) -> List[str]:
    """Rewrite legacy aliases to canonical Config flags; reject conflicting double writes (equal
    values pass, different values fail) before argparse can silently let the last occurrence win."""
    available = set(available_fields)
    owner_of = dict(field_owners or {})
    aliases: Dict[str, str] = {}
    for alias, target in CLI_ALIASES.items():
        if target not in available:
            continue
        # A legacy spelling that is itself a live field of a DIFFERENT Config than its target is that
        # Config's own flag, not a rename (--loss_type is the GRPO objective, --loss the loss); same-owner pairs still fold.
        source_owner = owner_of.get(alias)
        if source_owner is not None and owner_of.get(target) is not source_owner:
            continue
        aliases[alias] = target
    for alias, (target, _, _) in _BOOLEAN_ALIASES.items():
        if target in available:
            aliases[alias] = target

    result: List[str] = []
    seen: Dict[str, Tuple[str, object]] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith('--'):
            result.append(token)
            index += 1
            continue

        option, value, has_value, next_index = _scan_flag(argv, index)
        source = option.replace('-', '_')
        target = aliases.get(source, source)
        if source in _BOOLEAN_ALIASES and source in aliases:
            _, on_value, off_value = _BOOLEAN_ALIASES[source]
            enabled = True if value is None else _boolean_value(value)
            if enabled is None:
                raise ValueError(f'--{option} expects a boolean value, got {value!r}.')
            if enabled:
                value, has_value = on_value, True
            elif off_value is None:
                index = next_index
                continue
            else:
                value, has_value = off_value, True

        if target in available and (source in aliases or source == target):
            comparable = True if not has_value else _comparable_value(value)
            previous = seen.get(target)
            if previous is not None and previous[1] != comparable:
                raise ValueError(
                    f'Conflicting values for --{target}: --{previous[0]}={previous[1]!r} and '
                    f'--{option}={comparable!r}. Use one spelling or pass the same value.')
            seen[target] = (option, comparable)

        result.append(f'--{target}')
        if has_value:
            result.append(value)
        index = next_index
    return result


def _merged_parse_class(config_classes: Sequence[Type],
                        field_owners: Optional[Dict[str, Type]] = None) -> Tuple[Type, Dict[str, Type], Dict[str, Type]]:
    """Merge Configs into one argparse dataclass; a duplicate field must be pinned via ``field_owners``,
    else the value's destination depends on the Config listing order. Returns the class plus the
    first-declarer and effective-owner maps that ``parse_configs`` feeds to its consumers.
    """
    owners = dict(field_owners or {})
    present = set(config_classes)
    specs = []
    seen = {}
    declared_by: Dict[str, List[Type]] = {}
    first_owner: Dict[str, Type] = {}
    for config_class in config_classes:
        hints = get_type_hints(config_class)
        for config_field in fields(config_class):
            name = config_field.name
            first_owner.setdefault(name, config_class)
            if name in seen:
                declared_by[name].append(config_class)
                preferred = owners.get(name)
                if preferred is None:
                    raise ValueError(
                        f'CLI field {name!r} is declared by more than one Config '
                        f'({", ".join(cls.__name__ for cls in declared_by[name])}) with no owner pinned. A '
                        f'duplicate flag spelling must resolve to exactly one Config: pin it in this '
                        f"command's field_owners, or rename the field so it is unique across Configs.")
                if preferred not in present:
                    raise ValueError(
                        f'CLI field {name!r} is declared by more than one Config '
                        f'({", ".join(cls.__name__ for cls in declared_by[name])}) and pinned to owner '
                        f'{preferred.__name__}, which is not among this command\'s Configs. Fix the stale '
                        f'field_owners pin (or add {preferred.__name__} to the parsed Configs).')
                if preferred is config_class:
                    specs[seen[name]] = None
                else:
                    continue
            else:
                declared_by[name] = [config_class]
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
    merged = make_dataclass('MergedCliConfig', [spec for spec in specs if spec is not None])
    owner_of = {name: owners.get(name, cls) for name, cls in first_owner.items()}
    return merged, first_owner, owner_of


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
            # Keep the portable id alongside the resolved local dir: the swap happens even when the
            # snapshot turns out to carry no args.json, and model family matching reads the id's
            # basename (a snapshot dir's basename -- 'master', a hash -- matches nothing).
            if not isinstance(owner, list) and key == 'model' and resolved != source:
                setattr(owner, '_model_id_or_path', source)
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
    from packaging import version
    # model_type was reshaped after v4, so only trust an args.json written by 4.0.0.dev or newer.
    swift_version = old_args.get('swift_version')
    if swift_version is None or version.parse(swift_version) < version.parse('4.0.0.dev'):
        load_fields['model_config'].remove('model_type')
    for config_name, names in _CHECKPOINT_FORCE_LOAD_FIELDS.items():
        config = mapped.get(config_name)
        if config is None:
            continue
        for entry in names:
            key, field_name = entry if isinstance(entry, tuple) else (entry, entry)
            if old_args.get(key) is not None:
                setattr(config, field_name, old_args[key])
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

    ``field_owners`` selects which Config owns a duplicate CLI spelling; other Configs retain their
    own defaults, and each Config records which of its fields were explicit on the command line.
    """
    effective_argv = resolve_argv(argv)
    merged_class, first_owner, owner_of = _merged_parse_class(config_classes, field_owners)
    owners = dict(field_owners or {})
    effective_argv = normalize_argv(
        effective_argv, [config_field.name for config_field in fields(merged_class)], field_owners=owner_of)
    explicit_fields = flag_names(effective_argv)
    with _patch_type_hints():
        parser = HfArgumentParser(merged_class)
    merged, remaining = parser.parse_args_into_dataclasses(effective_argv, return_remaining_strings=True)
    values = vars(merged)

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
    # The checkpoint restore runs even with unrecognized args left over: they are either about to be
    # refused (strict) or deliberately tolerated (--allow_unknown_args), and neither case changes what
    # a checkpoint's args.json should fill back in.
    _restore_checkpoint_args(configs, default_load_args=load_args_default)
    for config in configs:
        _restore_config_types(config)
    return configs, remaining


#: Parser-level meta-flag (not backed by any Config field) that relaxes strict rejection of unknown args.
_ALLOW_UNKNOWN_ARGS_FLAG = 'allow_unknown_args'


def _strip_allow_unknown_args(argv: Sequence[str]) -> Tuple[List[str], bool]:
    """Pull the ``--allow_unknown_args`` meta-flag out of argv, returning ``(cleaned_argv, enabled)``.

    It carries no Config field -- it toggles parser behavior -- so it must be consumed here, else
    HfArgumentParser would drop it into the unrecognized remainder it exists to relax.
    """
    flag = f'--{_ALLOW_UNKNOWN_ARGS_FLAG}'
    cleaned: List[str] = []
    enabled = False
    i, n = 0, len(argv)
    while i < n:
        token = argv[i]
        if token == flag:
            # Bare presence enables it; a following boolean literal is its explicit value (other
            # tokens are NOT consumed: only a boolean literal is unambiguously this flag's value).
            if i + 1 < n and _boolean_value(argv[i + 1]) is not None:
                enabled = _boolean_value(argv[i + 1])
                i += 2
            else:
                enabled = True
                i += 1
            continue
        if token.startswith(f'{flag}='):
            value = _boolean_value(token.split('=', 1)[1])
            enabled = True if value is None else value
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned, enabled


def parse_configs_strict(
    config_classes: Sequence[Type],
    argv: Optional[List[str]] = None,
    *,
    command: str,
    field_owners: Optional[Dict[str, Type]] = None,
    load_args_default: bool = False,
    allow_unknown_args: bool = False,
) -> list:
    """Parse Configs and reject every unrecognized argument; ``--allow_unknown_args`` (or the
    caller's flag) downgrades that to a warning naming the flags no Config claimed."""
    effective_argv, flag_enabled = _strip_allow_unknown_args(resolve_argv(argv))
    configs, remaining = parse_configs(
        config_classes, effective_argv, field_owners=field_owners, load_args_default=load_args_default)
    if remaining:
        if allow_unknown_args or flag_enabled:
            logger.warning(f'Ignoring unrecognized arguments for {command} (no Config claims them, so they '
                           f'are dropped rather than applied): {remaining}.')
        else:
            raise ValueError(f'Unrecognized arguments for {command}: {remaining}. '
                             f'Pass --allow_unknown_args to tolerate them with a warning instead of failing.')
    return configs
