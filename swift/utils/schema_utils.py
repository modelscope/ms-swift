# Copyright (c) ModelScope Contributors. All rights reserved.
"""Helpers for the tool schemas carried by the ``tools`` column of a dataset."""
from typing import Any, Dict, Tuple

# The columns holding tool schemas: `tools` and the prefixed counterparts that
# `RowPreprocessor.standard_keys` derives from it, mirroring the message columns.
TOOL_KEYS: Tuple[str, ...] = ('tools', 'rejected_tools', 'positive_tools', 'negative_tools')

# JSON Schema keywords whose value maps a user defined name to a sub-schema.
_SCHEMA_MAP_KEYS = ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas', 'dependencies')
# Keywords where a literal ``null`` is a legal value.
_NULL_VALUE_KEYS = frozenset({'default', 'const'})
# Keywords where ``null`` is only legal as an item of the array value. Their items are
# literal data, so the arrays are kept verbatim and never recursed into.
_NULL_ITEM_KEYS = frozenset({'enum', 'examples'})


def remove_arrow_padding(schema: Any) -> Any:
    """Return a copy of ``schema`` without the ``null`` fields added by Arrow alignment.

    Tool schemas are heterogeneous nested dicts, but Arrow stores a column as a single
    struct type and aligns every row to the union of all its fields. The keys a tool never
    defined are then read back as ``None``: a tool defining only ``temperature`` arrives as
    ``{'temperature': {...}, 'query': None, ...}`` with one foreign entry per parameter
    used anywhere else in the dataset. Those entries leak into the system prompt and make
    agent templates raise on the null definitions.

    A ``null`` is only meaningful as the value of ``default``/``const``, as an item of an
    ``enum``/``examples`` array, or inside an ``x-`` prefixed annotation. Anywhere else it
    is padding and gets dropped. The input is never mutated.
    """
    if isinstance(schema, list):
        return [remove_arrow_padding(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in _NULL_VALUE_KEYS or key.startswith('x-'):
            cleaned[key] = value  # literal data, keep verbatim
        elif key in _NULL_ITEM_KEYS:
            if value is not None:
                cleaned[key] = value  # `null` items stay, a `null` array does not
        elif value is None:
            continue
        elif key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            cleaned[key] = {name: remove_arrow_padding(item) for name, item in value.items() if item is not None}
        else:
            cleaned[key] = remove_arrow_padding(value)
    return cleaned
