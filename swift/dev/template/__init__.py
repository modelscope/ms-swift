"""Dev-native template registry, family implementations and the single Template class they extend.

Public surface:

* :class:`Template` -- encoding, row preparation, media collation and twinkle's rollout contract.
* :class:`EncodeIntent` and the presets ``TRAIN`` / ``INFER`` / ``ENGINE`` / ``SCORE``.
* Inputs: ``SampleInput``; ``PreferenceSample`` / ``KTOSample`` select preference encoding.
* Outputs: ``EncodedRow`` or an explicit preference result mapping.
"""

from __future__ import annotations

from swift.dev.template_contract import (
    CompletionEncoding,
    DecodeContext,
    DecodedResponse,
    EncodedRow,
    KTOSample,
    MediaBundle,
    PreferenceSample,
    PromptEncoding,
)
from .base import MaxLengthError, Template
from .constant import TemplateCapability
from .conventions import NO_SHIFT_TASK_TYPES, is_bookkeeping_key
from .inputs import SampleInput, StdTemplateInputs, TemplateInputs
from .intent import ENGINE, INFER, SCORE, TRAIN, EncodeIntent
from .media import load_file
from .meta import TemplateMeta
from .register import register_template

__all__ = [
    "Template",
    "EncodeIntent",
    "TRAIN",
    "INFER",
    "ENGINE",
    "SCORE",
    "MaxLengthError",
    "load_file",
    "NO_SHIFT_TASK_TYPES",
    "is_bookkeeping_key",
    "MediaBundle",
    "PromptEncoding",
    "CompletionEncoding",
    "DecodedResponse",
    "DecodeContext",
    "SampleInput",
    "StdTemplateInputs",
    "TemplateInputs",
    "EncodedRow",
    "PreferenceSample",
    "KTOSample",
    "TemplateCapability",
    "TemplateMeta",
    "register_template",
]
