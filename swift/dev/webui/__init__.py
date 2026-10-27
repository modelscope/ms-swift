# Copyright (c) ModelScope Contributors. All rights reserved.
"""Declarative UI adapter layer for the workflow canvas.

Thin on purpose: schemas, descriptors and the registry are pure declarations; the
export turns them into the JSON the frontend consumes; the adapter maps a canvas
graph back onto real builders. No training logic lives here, and the components
in twinkle / swift.dev carry no knowledge of this package.
"""
from .adapter import CompileStep, EdgeSpec, GraphCompiler, GraphValidationError, NodeSpec, compile_graph
from .descriptor import ComponentDescriptor, ParamSpec, PortSpec
from .registry import REGISTRY, validate_registry
from .schema import CHANNEL_SCHEMAS, ChannelField, ChannelSchema

__all__ = [
    'CHANNEL_SCHEMAS',
    'CompileStep',
    'EdgeSpec',
    'GraphCompiler',
    'GraphValidationError',
    'NodeSpec',
    'compile_graph',
    'ChannelField',
    'ChannelSchema',
    'ComponentDescriptor',
    'ParamSpec',
    'PortSpec',
    'REGISTRY',
    'validate_registry',
]
