# Copyright (c) ModelScope Contributors. All rights reserved.
"""Export the registry and channel schemas as static JSON for the frontend.

Usage: ``python -m swift.dev.webui.export [out.json]`` (stdout when omitted).
The frontend treats the dumped file as build input; the Python registry stays
the single source of truth.
"""
from __future__ import annotations

import dataclasses
import json
import sys

from .execute import _LOADER_RENAMES, _RL_RENAMES
from .registry import REGISTRY, validate_registry
from .schema import CHANNEL_SCHEMAS

#: The Config classes the executor fills from canvas params. A registry param name is
#: "consumable" only if it lands on one of these (directly, via a rename, or through a
#: single-node construction path) -- anything else is a silent-failure seam.
_CONFIG_CLASSES = (
    'CheckpointConfig', 'ConvertConfig', 'DatasetConfig', 'DeployConfig',
    'DistributedConfig', 'EvalConfig', 'GenerationConfig', 'InferConfig',
    'LoggingConfig', 'ModelConfig', 'QuantizeConfig', 'RewardConfig', 'RLConfig',
    'RolloutConfig', 'RuntimeConfig', 'TemplateConfig', 'TrainConfig', 'TunerConfig',
)

#: Canvas spellings the executor routes to a differently-named Config field. The two
#: routing tables live in execute.py (the consumer); the two spellings below are
#: rewritten inline in its fill path (train_type->tuner, mini_micro_batch split) and
#: hub push (hub->use_hf, private->hub_private_repo).
_RENAMES = ({param for (_node_type, param) in _RL_RENAMES}
            | set(_LOADER_RENAMES)
            | {'train_type', 'mini_micro_batch', 'hub', 'private',
               'infer_backend',  # deploy -> DeployConfig.sampler_type
               'ema'})  # frozen_model ref-wiring -> rl.sync_ref_model (role-aware routing)

#: Params consumed by a single-node run's object construction, not by any Config.
_CONSTRUCTION_ONLY = {
    'tools',    # toolset: resolve_tool_plugins(names) via the plugin registry
    'base_url', 'model',  # api_model: APISampler/OpenAI(base_url, api_key)
    'name',     # code_component: the entry name in the user's code blob
    'run_script',  # external_rollout: the agent program to launch
    'kind',     # sandbox_env: LocalEnv vs AgentEnv
}

#: Declared for a planned seam, not yet consumed by any execution path. Each name
#: here was individually reviewed. Adding a name here is a decision, not a reflex:
#: a typo will NOT be in this set. (The placement tier, EMA shadow, signal form and
#: the driver-derived 'auto' values were either wired up or removed from the
#: registry; challenger synthesis remains the one planned seam.)
_DECLARED_ONLY = {
    'theme',
}


def validate_param_names() -> None:
    """Fail when a registry param name reaches no real consumer.

    A param that hits no Config field (directly, via a rename, or through a construction
    path) is a silent-failure seam: the canvas accepts the value and the executor drops
    it. This guard moves that discovery from runtime to export time.
    """
    import swift.dev.config as config_pkg
    known: set[str] = set()
    for cls_name in _CONFIG_CLASSES:
        cls = getattr(config_pkg, cls_name, None)
        if cls is None:
            raise ValueError(f'unknown Config class in the export check: {cls_name}')
        known.update(f.name for f in dataclasses.fields(cls))
    known |= _RENAMES | _CONSTRUCTION_ONLY | _DECLARED_ONLY
    offenders = sorted(
        f'{d.type}.{p.name}'
        for d in REGISTRY.values()
        for p in d.params
        if p.name not in known
    )
    if offenders:
        raise ValueError(
            'registry params that reach no Config field, rename, or construction path '
            f'(-- a silent-failure seam; fix the name or add it to the routing tables): {offenders}')


def build_payload() -> dict:
    """The full declarative payload: components + channel contracts + roles."""
    validate_registry()
    validate_param_names()
    return {
        'channels': {kind: schema.to_dict() for kind, schema in CHANNEL_SCHEMAS.items()},
        'components': {d.type: d.to_dict() for d in REGISTRY.values()},
        'version': 1,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    payload = build_payload()
    if not argv:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    else:
        with open(argv[0], 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'wrote {argv[0]}: {len(payload["components"])} components, '
              f'{len(payload["channels"])} channels')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
