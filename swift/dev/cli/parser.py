"""Self-parse of argv straight into dev's atomic Configs.

The dev CLI used to lean on legacy ``SftArguments``/``ExportArguments``: parse the legacy surface,
then copy same-named fields onto the Configs. That bridge dragged in ``swift/arguments`` and left the
legacy ``__post_init__`` side effects (output_dir versioning, plugin import, hub login, ...) behind.
This module removes the bridge: ``HfArgumentParser`` builds argparse options directly from the Config
dataclasses, so the Config *is* the argument surface and its defaults are the single source of truth.

Two annotation shapes ``HfArgumentParser`` cannot introspect are rewritten in ``_patch_type_hints``;
everything else on the SFT/export Config sets parses directly (verified: the 7 SFT Configs share no
field name, so a single flat argparse namespace has no collisions).
"""
from __future__ import annotations
import os
import sys
from contextlib import contextmanager
from dataclasses import MISSING, field, fields, make_dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union, get_args, get_origin, get_type_hints

import json
from transformers import HfArgumentParser


@contextmanager
def _patch_type_hints():
    """Normalise container unions that ``HfArgumentParser`` cannot safely introspect.

    JSON-backed mapping fields are parsed as strings and converted by ``process_configs``. This also
    avoids a Python typing-cache edge case where importing other modules can reorder ``Union`` members
    and make transformers call ``isinstance(None, typing.Dict[...])``. The scalar CLI form of
    ``Union[str, List[str], None]`` is likewise represented as ``Optional[str]``; its list form remains
    available to programmatic Config construction.
    """
    from transformers import hf_argparser
    origin_get_type_hints = hf_argparser.get_type_hints
    str_list_none = Union[str, List[str], type(None)]

    def get_type_hints(*args, **kwargs):
        hints = origin_get_type_hints(*args, **kwargs)
        for k, v in hints.items():
            union_args = get_args(v)
            has_mapping = any(arg is dict or get_origin(arg) is dict for arg in union_args)
            if type(None) in union_args and has_mapping:
                hints[k] = Optional[str]
            elif v == str_list_none:
                hints[k] = Optional[str]
        return hints

    hf_argparser.get_type_hints = get_type_hints
    try:
        yield
    finally:
        hf_argparser.get_type_hints = origin_get_type_hints


def resolve_argv(argv: Optional[List[str]] = None) -> List[str]:
    """Resolve the effective argv, including the Ray worker hand-off override."""
    ray_args = os.environ.get('RAY_SWIFT_ARGS')
    if ray_args:
        argv = json.loads(ray_args)
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise TypeError('RAY_SWIFT_ARGS must be a JSON array of strings.')
    return list(sys.argv[1:] if argv is None else argv)


def flag_names(argv: Sequence[str]) -> set:
    """Return normalized ``--flag`` names present in argv."""
    return {token[2:].split('=', 1)[0].replace('-', '_') for token in argv if token.startswith('--')}


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


def parse_configs(
    config_classes: Sequence[Type],
    argv: Optional[List[str]] = None,
    *,
    field_owners: Optional[Dict[str, Type]] = None,
) -> Tuple[list, List[str]]:
    """Parse argv into Config instances, including Config sets with duplicate field names.

    ``field_owners`` selects which Config owns a duplicate CLI spelling. Other Configs retain their
    own defaults; command-specific post-processing may deliberately mirror the parsed value when one
    flag controls more than one runtime component.
    """
    effective_argv = resolve_argv(argv)
    merged_class = _merged_parse_class(config_classes, field_owners)
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
        configs.append(config_class(**kwargs))
    return configs, remaining


def parse_configs_strict(
    config_classes: Sequence[Type],
    argv: Optional[List[str]] = None,
    *,
    command: str,
    field_owners: Optional[Dict[str, Type]] = None,
) -> list:
    """Parse Configs and reject every unrecognized argument."""
    configs, remaining = parse_configs(config_classes, argv, field_owners=field_owners)
    if remaining:
        raise ValueError(f'Unrecognized arguments for {command}: {remaining}.')
    return configs
