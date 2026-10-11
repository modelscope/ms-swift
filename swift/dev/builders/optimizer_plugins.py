# Copyright (c) ModelScope Contributors. All rights reserved.
"""The ``param_groups`` extension point: plugins that contribute optimizer param-group rules.

A param_groups plugin neither builds the optimizer nor touches the model -- it returns a list of
declarative, serializable grouping *rules* (grammar documented on
:func:`swift.dev.builders.optimizer.build_param_groups_spec`). dev composes the built-in rules (multimodal
per-tower learning rates, LoRA+) with whatever the user selects through
``TrainConfig.param_groups_plugins``, and hands the whole spec to twinkle's ``set_optimizer``, which
applies it worker-side on both backends (transformers: split the decay/no-decay groups by rule;
megatron: translate each rule into an mcore ``config_overrides`` entry).

Emitting *data* rather than a callable is what lets the customization cross the Ray boundary: the
driver only ever holds a PROXY model with no real parameters, so the matching (by name glob, param
``ndim``, or parent-module type) has to happen on the worker against the live model. twinkle therefore
stays model-agnostic -- it interprets the rules but hardcodes no tower or tuner naming.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin

__all__ = ['ParamGroupsPlugin', 'PARAM_GROUPS']


class ParamGroupsPlugin(SwiftPlugin):
    """Return extra optimizer param-group rules for a run.

    ``__call__`` receives the run's ``TrainConfig`` and ``TunerConfig`` (``None`` on a full-parameter
    run) and returns rules in the :func:`~swift.dev.builders.optimizer.build_param_groups_spec` grammar. The
    built-in per-tower / LoRA+ rules come first and the first rule matching a parameter wins, so a
    plugin appended here only claims parameters no built-in rule already matched.

    Example::

        @PluginRegistry.register('param_groups', 'no_decay_embeddings')
        class NoDecayEmbeddings(ParamGroupsPlugin):
            def __call__(self, *, train_config, tuner_config):
                return [{'match': {'module_type': 'Embedding'}, 'weight_decay': 0.0}]
    """

    def __call__(self, *, train_config: Any, tuner_config: Optional[Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


#: Declared at the bottom of the base's own module (never in an ``__init__.py``) so the kind exists
#: before any ``@PluginRegistry.register('param_groups', ...)`` decorator runs -- see the ordering note
#: in :mod:`swift.dev.utils.plugin`. This module is listed in ``PluginRegistry._BUILTIN_KIND_MODULES`` so a
#: user's ``--external_plugins`` file can register against the kind under a clean interpreter.
PARAM_GROUPS = PluginRegistry.register_kind('param_groups', ParamGroupsPlugin, config_field='param_groups_plugins')
