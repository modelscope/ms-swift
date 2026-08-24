# Copyright (c) ModelScope Contributors. All rights reserved.
"""Input-format converters: rewrite any raw dataset row into standard ``messages``.

Importing this package registers every built-in format, so :func:`get_converter` can find them.
"""
from .alpaca import AlpacaConverter
from .anthropic import AnthropicConverter
from .base import FORMAT_MAPPING, FormatConverter, Message, Messages, get_converter, list_formats, register_format
from .openai import OpenAIConverter
from .response import ResponseConverter

__all__ = [
    'FORMAT_MAPPING', 'FormatConverter', 'Message', 'Messages', 'get_converter', 'list_formats', 'register_format',
    'AlpacaConverter', 'AnthropicConverter', 'OpenAIConverter', 'ResponseConverter'
]
