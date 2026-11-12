# Copyright (c) ModelScope Contributors. All rights reserved.
"""The numeric compute plane for remote-execution stages.

Heavy forward payloads (logps / logits) must never materialise in the session
process: they flow between stages as infra references (LazyCollect), and
every numeric consumer -- response alignment, KL, shaping, advantage
estimation, combination, reduction -- runs on a twinkle signal component
placed on the numeric group, where the references are resolved. The session
only carries small control scalars back.

The session is ALWAYS ray: a reference passed as an argument is resolved by
the dispatch layer on the component's worker; the return value rides back as
ONE reference (``lazy_collect=True``), which downstream consumers (the next
component, ``train_step``'s small scalar read-back, the preview
materialisation) dereference where THEY run. The components are stateless
(every knob is a call argument), so one actor per class serves every node.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Tuple, Type

from .placement import NUMERIC_GROUP


def deref(value: Any) -> Any:
    """Resolve a maybe-lazy remote result HERE (never in the driver); pass others through."""
    from swift.dev.utils.scoring import collect
    return collect(value)


_components: Dict[Tuple[str, type], Any] = {}
_components_lock = threading.Lock()


def numeric_component(cls: Type[Any], remote_group: str = NUMERIC_GROUP) -> Any:
    """The process-resident actor of a stateless twinkle signal component (one per class per group).

    Built on first use under ``remote_group`` -- the CPU group the session bootstrap declared at
    ``twinkle.initialize`` time (the placement module's NUMERIC_GROUP, the single spelling of that
    name). A session started without the group fails on the first remote-execution stage.

    The lock is load-bearing: the DAG runs concurrently-ready stages in parallel threads, and a
    plain cache lets two simultaneous first uses BOTH construct the actor -- its name is
    deterministic (group + class + call site), so the second creation dies with 'actor already
    exists'.
    """
    key = (remote_group, cls)
    component = _components.get(key)
    if component is not None:
        return component
    with _components_lock:
        component = _components.get(key)
        if component is None:
            component = cls(remote_group=remote_group)
            _components[key] = component
        return component
