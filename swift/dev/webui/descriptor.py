# Copyright (c) ModelScope Contributors. All rights reserved.
"""Component declarations for the workflow canvas.

A descriptor states what a component is (math_doc), what it reads and writes
(ports), which knobs it takes (params) and where the real implementation lives
(factory). Descriptions live here, behaviour stays in twinkle / swift.dev -- the
gap between the two sides is absorbed entirely on this side, per the layering
decision: python components never carry UI concepts.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PortSpec:
    """One port. ``role`` is the connection guidance shown on the canvas
    (what wiring this port means semantically); ``many`` marks a symmetric
    collection port (e.g. teachers) that lawfully takes several in-edges;
    ``optional`` marks an input the component can run without (the wiring
    decides: ref is only needed with KL, traj only for preference losses)."""

    key: str
    kind: str
    role: str
    many: bool = False
    optional: bool = False

    def to_dict(self) -> dict:
        return {'key': self.key, 'kind': self.kind, 'role': self.role, 'many': self.many,
                'optional': self.optional}


@dataclass(frozen=True)
class ParamSpec:
    """One knob. ``options`` makes it an enum whose values are plain math names
    (estimator or loss families), so one descriptor serves a whole family."""

    name: str
    label: str
    default: str
    doc: str = ''
    options: tuple[str, ...] = ()
    group: str = ''

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'label': self.label,
            'default': self.default,
            'doc': self.doc,
            'options': list(self.options),
            'group': self.group,
        }


@dataclass(frozen=True)
class ComponentDescriptor:
    """The single source of truth for one canvas component type."""

    type: str
    label: str
    category: str
    math_doc: str
    ports_in: tuple[PortSpec, ...] = field(default=())
    ports_out: tuple[PortSpec, ...] = field(default=())
    params: tuple[ParamSpec, ...] = field(default=())
    factory: str = ''
    #: Construction point when deploy_mode=server (the client-side handle).
    #: Empty means the component always constructs on the training side.
    client_factory: str = ''

    def port(self, key: str) -> PortSpec:
        for spec in self.ports_in + self.ports_out:
            if spec.key == key:
                return spec
        raise KeyError(f'component {self.type!r} has no port {key!r}')

    def to_dict(self) -> dict:
        return {
            'type': self.type,
            'label': self.label,
            'category': self.category,
            'math_doc': self.math_doc,
            'ports_in': [p.to_dict() for p in self.ports_in],
            'ports_out': [p.to_dict() for p in self.ports_out],
            'params': [p.to_dict() for p in self.params],
            'factory': self.factory,
            'client_factory': self.client_factory,
        }
