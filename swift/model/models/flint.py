# Copyright (c) ModelScope Contributors. All rights reserved.
"""Flint (Recursive).

transformers>=5.21 ships the architecture in `transformers.models.flint`, including the checkpoint
conversion that turns the released `self_attn.softmax_offset` and per-expert
`mlp.experts.{i}.{gate,up,down}_proj` weights into its `sinks` and stacked experts, so ms-swift only
registers the model type and its chat template here.
"""

from swift.template import TemplateType
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import register_model

register_model(
    ModelMeta(
        LLMModelType.flint,
        [
            ModelGroup([
                Model('RecursiveSI/flint', 'RecursiveSI/flint'),
            ]),
        ],
        template=TemplateType.flint,
        architectures=['FlintForCausalLM'],
        requires=['transformers>=5.21'],
    ))
