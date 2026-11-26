# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared encoding intents and adapters for swift input objects."""
from __future__ import annotations

from twinkle.template.contract import ENGINE, INFER, SCORE, TRAIN, EncodeIntent, EngineBackend, resolve_intent
from twinkle.template.contract.encoding import check_generation_prompt as _check_generation_prompt

from .inputs import TemplateInputs


def check_generation_prompt(sample, add_generation_prompt):
    if isinstance(sample, TemplateInputs):
        sample = {'messages': sample.chosen.messages}
    return _check_generation_prompt(sample, add_generation_prompt)
