# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resource bindings: built-once native objects (model, template, dataset...).

Every binding follows the same shape: collect the Config-class spellings from
the canvas params (names ARE the field names; only the few combined canvas
spellings split here), construct the native object through the EXISTING
builders, keep it resident. Nothing native learns about the canvas.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

from ..component import (ComponentError, ResourceComponent, StageComponent,
                         load_code_component)


def _fill(cls: Any, kv: Dict[str, str], node_id: str) -> Any:
    """Instantiate one Config dataclass from string params keyed by field name.

    A param whose name is not a field fails loudly here (name the seam), never
    silently drops. Values are coerced by the field type; empty string means
    "unset" for Optional fields.
    """
    import typing

    hints = typing.get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    known: Dict[str, Any] = {}
    for key, value in kv.items():
        if key not in fields:
            raise ComponentError(f'{node_id}: param {key!r} is not a field of {cls.__name__}')
        if value == '' and typing.get_origin(hints.get(key)) is typing.Union:
            continue
        known[key] = _coerce(value, hints.get(key, str), node_id, key)
    return cls(**known)


def _coerce(value: str, hint: Any, node_id: str, key: str) -> Any:
    import typing

    origin = typing.get_origin(hint)
    if origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = args[0] if args else str
        origin = typing.get_origin(hint)
    if hint is bool:
        if value.lower() not in ('true', 'false'):
            raise ComponentError(f'{node_id}.{key}: expected true/false, got {value!r}')
        return value.lower() == 'true'
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    if origin in (list, typing.List) or hint in (typing.List, list):
        parts = [p for chunk in value.split(',') for p in chunk.split() if p]
        inner = typing.get_args(hint)
        inner_hint = inner[0] if inner else str
        return [_coerce(p, inner_hint, node_id, key) for p in parts]
    return value  # str / Literal / Any: as-is


def _configs():
    """The dev Config namespace, imported lazily (the session process may run
    before any training import)."""
    import swift.dev.config as config_pkg
    return config_pkg


def build_template_from(tokenizer_id: str, template_kv: Optional[dict], node_id: str) -> Any:
    """The encoding rule for ONE tokenizer source: a resident real template.

    The processor loads WITHOUT the model (``load_model=False``), so the rule
    exists before any GPU model does. ``load_model_processor`` returns
    ``(model_or_None, processor)`` -- only the processor is consumed here.
    A missing source fails loudly by name."""
    from swift.dev.builders import build_template, load_model_processor
    model_id = str(tokenizer_id or '').strip()
    if not model_id:
        raise ComponentError(f'{node_id}: tokenizer source is required -- '
                             'params.tokenizer names the model whose tokenizer loads')
    kv = dict(template_kv or {})
    try:
        template_config = _configs().TemplateConfig(**kv)
        model_config = _configs().ModelConfig(model=model_id)
    except TypeError as exc:
        raise ComponentError(f'{node_id}: {exc}') from None
    _model_or_none, processor = load_model_processor(model_config, load_model=False)
    return build_template(template_config, processor)


def encode_rows(template: Any, data: Any) -> list:
    """One batch of rows / trajectories -> native encoded features.

    Dict rows (dataset rows, ``{'messages': ...}``) pass to ``template.encode``
    directly; trajectory objects are normalised to their message rows first --
    the same per-round re-encode the dev recipes run (RL rollouts are
    encode=False by design). Anything unrecognised fails loudly."""
    items = data if isinstance(data, list) else (
        (data or {}).get('rows', []) if isinstance(data, dict) else [])
    if items and not isinstance(items[0], dict):
        rows = [getattr(item, 'messages', None) for item in items]
        if any(row is None for row in rows):
            raise ComponentError(
                'cannot encode: items are neither message rows nor trajectories')
        items = [{'messages': row} for row in rows]
    encode = getattr(template, 'encode', None)
    if not callable(encode):
        raise ComponentError(f'the template {template!r} exposes no encode()')
    return encode(items)


class ModelComponent(ResourceComponent):
    """The trainable f_theta: build_model over ModelConfig/DistributedConfig."""

    type = 'model'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from swift.dev.builders.model import build_model
        config_pkg = _configs()
        kv = {k: str(v) for k, v in self.params.items()
              if k in {'model', 'torch_dtype', 'model_revision', 'load_device', 'seed', 'fuse_x', 'low_cpu_mem_usage'}}
        model_config = _fill(config_pkg.ModelConfig, kv, self.node_id)
        dist_config = _fill(config_pkg.DistributedConfig,
                            {'nproc_per_node': '1', 'mode': 'local'}, self.node_id)
        tuner_kv = {k: str(v) for k, v in self.params.items()
                    if k in {'train_type', 'adapters', 'lora_rank', 'lora_alpha', 'lora_dropout',
                             'use_rslora', 'use_dora', 'target_modules', 'modules_to_save'}}
        if 'train_type' in tuner_kv:
            tuner_kv['tuner'] = tuner_kv.pop('train_type')
        tuner_config = _fill(config_pkg.TunerConfig, tuner_kv, self.node_id) if tuner_kv \
            else None
        if tuner_config is not None and tuner_config.tuner == 'full':
            tuner_config = None
        # train_config carries NATIVE TrainConfig field spellings as a dict;
        # _fill fails loudly on a name that is not a real field.
        train_kv = {k: str(v) for k, v in dict(self.params.get('train_config') or {}).items()}
        train_config = _fill(config_pkg.TrainConfig, train_kv, self.node_id) if train_kv else None
        self.obj = build_model(model_config, dist_config, train_config=train_config,
                               tuner_config=tuner_config)
        return self.obj


class DatasetComponent(ResourceComponent):
    """The data distribution: rows + prompts off the real loader path."""

    type = 'dataset'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from swift.dev.recipe.grpo.run_grpo import prompt_rows_from_dataset
        kv = {k: str(v) for k, v in self.params.items()
              if k in {'dataset', 'packing', 'packing_strategy', 'streaming', 'lazy_tokenize',
                       'split_dataset_ratio', 'dataloader_num_workers', 'train_dataloader_shuffle',
                       'dataloader_drop_last', 'dataset_seed'}}
        dataset_config = _fill(_configs().DatasetConfig, kv, self.node_id)
        rows, prompts, extras = prompt_rows_from_dataset(dataset_config)
        self.obj = {'rows': rows, 'prompts': prompts, 'extras': extras, 'config': dataset_config}
        return self.obj


class SamplerComponent(ResourceComponent):
    """A resident inference engine over an explicit model id.

    The tokenizer/template comes from the SAME id (one source of truth); the
    engine and its knobs ride ``build_sampler``'s real signature
    (``ModelConfig`` + backend + ``engine_args`` + template)."""

    type = 'sampler'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from swift.dev.builders.sampler import build_sampler
        model_id = str(self.params.get('model') or '').strip()
        if not model_id:
            raise ComponentError(f'{self.node_id}: sampler needs params.model (the model id)')
        template = build_template_from(
            model_id, dict(self.params.get('template_config') or {}), self.node_id)
        cfg = dict(self.params.get('model_config') or {})
        cfg['model'] = model_id
        try:
            model_config = _configs().ModelConfig(**cfg)
        except TypeError as exc:
            raise ComponentError(f'{self.node_id}: model_config: {exc}') from None
        self.obj = build_sampler(
            model_config,
            backend=str(self.params.get('engine', 'vllm') or 'vllm'),
            engine_args=dict(self.params.get('engine_args') or {}) or None,
            template=template)
        return self.obj


class RolloutComponent(ResourceComponent):
    """The multi-turn episode engine: a native MultiTurnRollout binding.

    Tools arrive as a wired ToolManager, or through the ENV that executes them
    (``env.tool_manager()``) -- both wired with an env that has tools is a
    conflict, refused rather than merged."""

    type = 'rollout'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
        sampler = inputs.get('sampler')
        if sampler is None:
            raise ComponentError(f'{self.node_id}: rollout needs a wired sampler')
        tools = inputs.get('tools')
        env = inputs.get('env')
        env_tools = bool(env is not None and env.tools())
        if tools is not None and env_tools:
            raise ComponentError(
                f'{self.node_id}: rollout takes ONE tool source -- a wired ToolManager or '
                'the env that executes them, not both')
        tool_manager = tools if tools is not None else (env.tool_manager() if env_tools else None)
        template = build_template_from(
            str(self.params.get('tokenizer') or ''),
            dict(self.params.get('template_config') or {}), self.node_id)
        cfg = dict(self.params.get('rollout_config') or {})
        cfg.pop('max_turns', None)
        gen = dict(self.params.get('generation_config') or {})
        sampling_params = None
        if gen:
            from twinkle.data_format.sampling import SamplingParams
            try:
                sampling_params = SamplingParams(**gen)
            except TypeError as exc:
                raise ComponentError(f'{self.node_id}: generation_config: {exc}') from None
        try:
            self.obj = MultiTurnRollout(
                sampler, template=template, tool_manager=tool_manager,
                sampling_params=sampling_params,
                max_turns=int(self.params.get('max_turns', '8') or 8), **cfg)
        except TypeError as exc:
            raise ComponentError(f'{self.node_id}: rollout_config: {exc}') from None
        return self.obj


class GroupRuleComponent(ResourceComponent):
    """The complete-group admission rule: the native GroupAdmissionPolicy.

    ``zero_variance`` turns on the reward-std gate (a group whose rewards do
    not vary carries no learning signal); every other threshold arrives via
    the typed ``kwargs``. No forward is ever run here -- baselines belong to
    the advantage estimator."""

    type = 'group_rule'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle.advantage.group_admission import GroupAdmissionConfig, GroupAdmissionPolicy
        kwargs = dict(self.params.get('kwargs') or {})
        if self.params.get('resample', 'none') == 'zero_variance':
            kwargs.setdefault('min_reward_std', 1e-6)
        try:
            config = GroupAdmissionConfig(**kwargs)
        except TypeError as exc:
            raise ComponentError(
                f'{self.node_id}: admission kwargs {sorted(kwargs)} rejected: {exc}') from exc
        self.obj = GroupAdmissionPolicy(config)
        return self.obj


class ToolsetComponent(ResourceComponent):
    """The real tool plugins: module sources loaded through twinkle's loader.

    A plugin module's module-level Tool instances ARE the payload -- the
    declaration and the implementation are the same thing, exactly like the
    plugin path the recipes use. Runtime binding to an environment lease is
    the consumer's business (rollout / challenger)."""

    type = 'toolset'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle.utils import load_module
        from twinkle_agentic.tools import ToolManager
        tools = []
        for source in (self.params.get('plugins') or []):
            module = load_module(str(source))
            found = [value for value in vars(module).values()
                     if callable(getattr(value, 'tool_info', None)) and not isinstance(value, type)]
            if not found:
                raise ComponentError(
                    f'{self.node_id}: plugin {source!r} defines no tool instance '
                    '(a tool is a module-level object exposing tool_info())')
            tools.extend(found)
        self.obj = ToolManager(tools)
        return self.obj


class SandboxEnvComponent(ResourceComponent):
    """The tool-call sandbox; kind picks the native backend."""

    type = 'sandbox_env'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle_agentic.envs import LocalEnv
        kind = self.params.get('kind', 'local')
        if kind != 'local':
            raise ComponentError(
                f'{self.node_id}: sandbox kind {kind!r} arrives with its native '
                'builder; only local is bound here')
        self.obj = LocalEnv()
        return self.obj


class PolicyEndpointComponent(ResourceComponent):
    """A local OpenAI-shaped endpoint over the serving sampler."""

    type = 'policy_endpoint'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle_agentic.rollout.endpoint import PolicyEndpoint
        sampler = inputs.get('sampler')
        if sampler is None:
            raise ComponentError(f'{self.node_id}: policy_endpoint needs a wired sampler')
        self.obj = PolicyEndpoint(sampler, port=int(self.params.get('port', '8000') or 8000))
        return self.obj


class ApiEndpointComponent(ResourceComponent):
    """An external OpenAI-compatible connection."""

    type = 'api_endpoint'

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle_agentic.protocol.openai import OpenAI
        base_url = self.params.get('base_url', '') or None
        model = self.params.get('model', '') or 'unknown'
        # Local compatible endpoints take any placeholder key; the SDK refuses
        # to construct without one. No request is sent at build time.
        self.obj = OpenAI(model=model, base_url=base_url, api_key='EMPTY')
        return self.obj


class CodeComponent(ResourceComponent):
    """A user code blob as a resident callable. The blob IS the payload; the
    entry contract (fn / last def) matches the plugin path."""

    type = 'code_component'

    def build(self, inputs: Dict[str, Any]) -> Any:
        code = self.params.get('code', '') or getattr(self, 'code', '')
        if not code:
            raise ComponentError(f'{self.node_id}: this code component has no function body')
        self.obj = load_code_component(code, self.params.get('name', 'fn'))
        return self.obj
