# Copyright (c) ModelScope Contributors. All rights reserved.
"""Declarative UI adapter layer for the workflow canvas.

Thin on purpose: schemas, descriptors and the registry are pure declarations;
the graph module validates and plans; the loop and session drive the resident
execution; the components package binds real twinkle / swift.dev objects. No
training logic lives here, and the components in twinkle / swift.dev carry no
knowledge of this package.
"""
from .component import (COMPONENT_CLASSES, ComponentError, IteratorComponent,
                         ResourceComponent, StageComponent, TaskComponent,
                         load_code_component, load_plugin_source, register_component)
from .descriptor import LIFECYCLES, ComponentDescriptor, ParamSpec, PortSpec
from .graph import EdgeSpec, FrameSpec, Graph, GraphValidationError, NodeSpec, parse_declaration
from .registry import REGISTRY, validate_registry
from .schema import ALLOWS, CHANNEL_SCHEMAS, ChannelField, ChannelProfile, ChannelSchema, allows

__all__ = [
    'ALLOWS',
    'CHANNEL_SCHEMAS',
    'COMPONENT_CLASSES',
    'ChannelField',
    'ChannelProfile',
    'ChannelSchema',
    'ComponentDescriptor',
    'ComponentError',
    'EdgeSpec',
    'FrameSpec',
    'Graph',
    'GraphValidationError',
    'LIFECYCLES',
    'NodeSpec',
    'ParamSpec',
    'PortSpec',
    'IteratorComponent',
    'ResourceComponent',
    'StageComponent',
    'TaskComponent',
    'REGISTRY',
    'allows',
    'load_code_component',
    'load_plugin_source',
    'parse_declaration',
    'register_component',
    'validate_registry',
]
