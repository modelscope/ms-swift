# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resource bindings: built-once native objects (model, template, dataset...).

Every binding follows the same shape: collect the Config-class spellings from
the canvas params (names ARE the field names; only the few combined canvas
spellings split here), construct the native object through the EXISTING
builders, keep it resident. Nothing native learns about the canvas.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional

from swift.dev.utils.logger import get_logger

from ..component import (ComponentError, ResourceComponent, StageComponent,
                         load_code_component)
from twinkle.loss import Loss

logger = get_logger()


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


def _coerce(value: Any, hint: Any, node_id: str, key: str) -> Any:
    import typing

    origin = typing.get_origin(hint)
    if origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = args[0] if args else str
        origin = typing.get_origin(hint)
    # v3 typed values arrive as native JSON: a list stays a list, a number
    # stays a number. Only string spellings (the legacy panel form) go
    # through the parse path below -- str(['x']) would smuggle the repr
    # brackets into the value and corrupt ids and paths.
    if origin in (list, typing.List) or hint in (typing.List, list):
        if isinstance(value, list):
            inner = typing.get_args(hint)
            inner_hint = inner[0] if inner else str
            return [_coerce(v, inner_hint, node_id, key) for v in value]
    elif isinstance(value, bool) and hint is bool:
        return value
    elif isinstance(value, int) and hint is int:
        return value
    elif isinstance(value, (int, float)) and hint is float:
        return float(value)
    if not isinstance(value, str):
        # Only the explicitly scalar hints mismatch loudly; str / Any /
        # Literal and unknown hints take the value as-is (the typed form
        # is the source of truth).
        if hint is int or hint is float or hint is bool:
            raise ComponentError(
                f'{node_id}.{key}: expected {hint.__name__}, got {type(value).__name__}')
        return value
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


def _reject_unknown_kwargs(cls: Any, kwargs: Dict[str, Any], node_id: str, label: str) -> None:
    """Fail loudly on a constructor kwarg no ``__init__`` in the MRO names.

    Some native constructors accept ``**kwargs`` and silently drop what they
    do not know (the GRPO loss family forwards nothing) -- a typo on the panel
    would then run on defaults without a trace. The guard collects the named
    parameters of every ``__init__`` in the class chain: a name absent
    everywhere is refused here, before the silent drop can happen. Catch-all
    parameters (``*args`` / ``**kwargs``) do not count -- their names are
    spelling artefacts, not settable fields."""
    import inspect
    named = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get('__init__')
        if init is None:
            continue
        named |= {param.name for param in inspect.signature(init).parameters.values()
                  if param.name not in ('self', 'args')
                  and param.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                         inspect.Parameter.VAR_KEYWORD)}
    unknown = set(kwargs) - named
    if unknown:
        raise ComponentError(
            f'{node_id}: {label} rejects kwargs {sorted(unknown)} '
            f'(constructor accepts {sorted(named) if named else "no kwargs"})')


def build_template_from(tokenizer_id: str, template_kv: Optional[dict], node_id: str,
                        model_kv: Optional[dict] = None) -> Any:
    """The encoding rule for ONE tokenizer source: a resident real template.

    The processor loads WITHOUT the model (``load_model=False``), so the rule
    exists before any GPU model does. ``load_model_processor`` returns
    ``(model_or_None, processor)`` -- only the processor is consumed here.
    ``model_kv`` carries identity fields the tokenizer load also needs (a
    locally built directory is ambiguous without an explicit ``model_type``);
    it must stay consistent with the model's own ModelConfig. A missing
    source fails loudly by name."""
    from swift.dev.builders import build_template, load_model_processor
    model_id = str(tokenizer_id or '').strip()
    if not model_id:
        raise ComponentError(f'{node_id}: tokenizer source is required -- '
                             'params.tokenizer names the model whose tokenizer loads')
    kv = dict(template_kv or {})
    extra = {k: v for k, v in dict(model_kv or {}).items() if k != 'model'}
    try:
        template_config = _configs().TemplateConfig(**kv)
        model_config = _configs().ModelConfig(model=model_id, **extra)
    except TypeError as exc:
        raise ComponentError(f'{node_id}: {exc}') from None
    _model_or_none, processor = load_model_processor(model_config, load_model=False)
    return build_template(template_config, processor)


def encode_rows(template: Any, data: Any) -> list:
    """One batch of rows / trajectories -> native encoded features.

    The native batch entry is ``batch_encode`` (list[Trajectory]). The batch
    arrives as a row list (a twinkle DataLoader batch) or as a dict exposing
    ``rows``. A rollout trajectory already carries its encoding (``input_ids``
    / ``labels``) -- that IS the authority (labels already shifted, aligned
    with ``rollout_logprobs``), so it is returned verbatim: re-encoding its
    messages would rebuild the labels from scratch and count turn-boundary
    tokens the ledger never put in the response, misaligning every token-level
    lane downstream. Anything unrecognised fails loudly."""
    items = data if isinstance(data, list) else (
        (data or {}).get('rows', []) if isinstance(data, dict) else [])
    if isinstance(data, dict) and not isinstance(data.get('rows', []), list):
        raise ComponentError(
            'cannot encode: a dict batch must expose a row list under "rows" '
            f'(got keys {sorted(data.keys())[:8]})')
    if items and not all(isinstance(item, dict) for item in items):
        raise ComponentError('cannot encode: items are neither message rows nor trajectories')
    if items and all(item.get('input_ids') is not None for item in items):
        return list(items)
    if any(item.get('input_ids') is not None for item in items):
        failed = [item.get('error') for item in items if item.get('input_ids') is None]
        raise ComponentError(
            f'cannot encode: {len(failed)} of {len(items)} trajectories carry no encoding '
            f'(a failed episode, e.g. {failed[0]!r}); re-encoding them would misalign the batch')
    encode = getattr(template, 'batch_encode', None)
    if not callable(encode):
        raise ComponentError(f'the template {template!r} exposes no batch_encode()')
    result = encode(items)
    return result if isinstance(result, list) else [result]


class ModelComponent(ResourceComponent):
    """The trainable f_theta: build_model over ModelConfig/DistributedConfig.

    The session is ALWAYS ray: mode is fixed here, and the placement rides
    the session's device plan -- the injected ``remote_group`` names the
    model's own DeviceGroup, the card spec sizes ``nproc_per_node``, and
    ``parallel_spec`` (explicit tp/pp/dp) shapes the mesh. The backend is a
    plain parameter: both tiers go through the SAME ``build_model`` entry
    with zero backend branches here; the megatron tier additionally passes
    its megatron/moe config groups through."""

    type = 'model'

    #: build 在并行阶段会首次加载的模块（含函数体内 lazy import 的入口）：
    #: 预热必须覆盖它们，否则 worker 的深链 import 会与主线程设备构建的
    #: transformers 首次加载交错（半初始化模块上的 from-import 失败）。
    IMPORTS = ('swift.dev.config', 'swift.dev.builders.model', 'swift.dev.builders.adapter',
               'transformers')

    def build(self, inputs: Dict[str, Any]) -> Any:
        from swift.dev.builders.model import build_model
        config_pkg = _configs()
        backend = str(self.params.get('backend', 'hf') or 'hf')
        model_kv = dict(self.params.get('model_config') or {})
        model_kv['model'] = self.params.get('model')
        model_config = _fill(config_pkg.ModelConfig, model_kv, self.node_id)
        dist_kv = dict(self.params.get('distributed_config') or {})
        dist_kv.update({'mode': 'ray',  # the session boots ray; there is no local form
                        'backend': backend,
                        'nproc_per_node': self._card_count(),
                        'parallel_spec': str(self.params.get('parallel_spec') or '') or None})
        dist_config = _fill(config_pkg.DistributedConfig, dist_kv, self.node_id)
        if not self.remote_group:
            raise ComponentError(
                f'{self.node_id}: a model needs its planned DeviceGroup -- '
                'run it through a booted session (the resource page picks the cards)')
        megatron_config = moe_config = None
        if backend == 'megatron':
            kv = dict(self.params.get('megatron_config') or {})
            megatron_config = _fill(config_pkg.MegatronConfig, kv, self.node_id) if kv else None
            kv = dict(self.params.get('moe_config') or {})
            moe_config = _fill(config_pkg.MoeConfig, kv, self.node_id) if kv else None
        elif self.params.get('megatron_config') or self.params.get('moe_config'):
            raise ComponentError(
                f'{self.node_id}: megatron_config/moe_config 仅在 backend=megatron 时生效')
        train_type = str(self.params.get('train_type') or 'lora')
        tuner_kv = dict(self.params.get('tuner_config') or {})
        if train_type != 'full':
            tuner_kv['tuner'] = train_type
        tuner_config = _fill(config_pkg.TunerConfig, tuner_kv, self.node_id) if tuner_kv else None
        # train_config carries NATIVE TrainConfig field spellings as a dict;
        # _fill fails loudly on a name that is not a real field.
        train_kv = dict(self.params.get('train_config') or {})
        train_config = _fill(config_pkg.TrainConfig, train_kv, self.node_id) if train_kv else None
        # The mesh is derived inside build_model from parallel_spec +
        # nproc_per_node (build_dp_mesh for transformers, the megatron bridge
        # for megatron) -- the same derivation every recipe uses, so the
        # webui and run_grpo cannot drift apart in placement.
        self.obj = build_model(model_config, dist_config, train_config=train_config,
                               tuner_config=tuner_config, megatron_config=megatron_config,
                               moe_config=moe_config, remote_group=self.remote_group)
        if tuner_config is not None:
            # The adapter install is a SEPARATE step after build_model (the same
            # apply_tuner every recipe runs); without it train_type='lora'
            # silently degrades to a full-parameter run. GA threads the
            # optimizer-group scale the recipes pass from TrainConfig.
            from swift.dev.builders.adapter import apply_tuner
            apply_tuner(
                self.obj, tuner_config,
                gradient_accumulation_steps=(
                    train_config.gradient_accumulation_steps if train_config else 1),
                task_type=model_config.task_type)
        if self.params.get('freeze'):
            # A frozen resident (reference / teacher / value anchor): the
            # driver-facing handle works on every placement, proxy included.
            self.obj.freeze_parameters(freeze_ratio=1.0)
        return self.obj

    def _card_count(self) -> int:
        """The declared card spec: the explicit list's length, else the count."""
        ids = self.params.get('gpu_ids') or []
        return len(ids) if ids else int(self.params.get('gpu_count') or 1)


class DatasetComponent(ResourceComponent):
    """The data distribution: rows + prompts off the real loader path."""

    type = 'dataset'

    #: swift.dev.dataset 是 load_prompt_only_rows 深链的 lazy 入口，其顶层
    #: 连带加载 transformers（swift.dev.utils.utils 的 from-import）——预热它
    #: 才能保证 worker 线程的 from transformers import 命中已加载模块。
    IMPORTS = ('swift.dev.config', 'swift.dev.recipe.grpo.run_grpo', 'swift.dev.dataset')

    def build(self, inputs: Dict[str, Any]) -> Any:
        from swift.dev.recipe.grpo.run_grpo import prompt_rows_from_dataset
        kv = {k: v for k, v in self.params.items()
              if k in {'dataset', 'packing', 'packing_strategy', 'streaming', 'lazy_tokenize',
                       'split_dataset_ratio', 'dataloader_num_workers', 'train_dataloader_shuffle',
                       'dataloader_drop_last', 'dataset_seed'}}
        dataset_config = _fill(_configs().DatasetConfig, kv, self.node_id)
        rows, prompts, extras = prompt_rows_from_dataset(dataset_config)
        self.obj = {'rows': rows, 'prompts': prompts, 'extras': extras, 'config': dataset_config}
        return self.obj


class SamplerComponent(ResourceComponent):
    """A resident inference engine over an explicit model id, DISAGGREGATED.

    The engine owns its planned DeviceGroup (the injected ``remote_group``)
    -- never a card shared with a trainer. The tokenizer/template comes from
    the SAME id (one source of truth); the parallel width is the declared
    ``parallel_spec`` (tp, default: one engine spanning every declared card),
    and ``engine_args`` keep any explicit override. The engine and its knobs
    ride ``build_sampler``'s real signature."""

    type = 'sampler'

    IMPORTS = ('swift.dev.config', 'swift.dev.builders', 'swift.dev.builders.sampler',
               'transformers')

    def build(self, inputs: Dict[str, Any]) -> Any:
        from swift.dev.builders.sampler import build_sampler
        from ..placement import sampler_tp
        model_id = str(self.params.get('model') or '').strip()
        if not model_id:
            raise ComponentError(f'{self.node_id}: sampler needs params.model (the model id)')
        if not self.remote_group:
            raise ComponentError(
                f'{self.node_id}: a sampler needs its planned DeviceGroup -- '
                'run it through a booted session (the resource page picks the cards)')
        template = build_template_from(
            model_id, dict(self.params.get('template_config') or {}), self.node_id,
            model_kv=dict(self.params.get('model_config') or {}))
        cfg = dict(self.params.get('model_config') or {})
        cfg['model'] = model_id
        try:
            model_config = _configs().ModelConfig(**cfg)
        except TypeError as exc:
            raise ComponentError(f'{self.node_id}: model_config: {exc}') from None
        engine = str(self.params.get('engine', 'vllm') or 'vllm')
        engine_args = dict(self.params.get('engine_args') or {})
        # The engine's shape mirrors its DeviceGroup: tp GPUs per worker,
        # cards/tp workers. Derived from the declared spec exactly as the
        # planner sized the group, so the engine and the group cannot
        # drift; an explicit engine_args entry still wins.
        cards = self._card_count()
        tp = sampler_tp(str(self.params.get('parallel_spec') or ''), cards)
        engine_args.setdefault('tensor_parallel_size', tp)
        replicas = cards // tp
        if replicas > 1:
            engine_args.setdefault('data_parallel_size', replicas)
        from twinkle import DeviceMesh
        self.obj = build_sampler(
            model_config,
            backend=engine,
            engine_args=engine_args or None,
            device_mesh=DeviceMesh.from_sizes(dp_size=replicas),
            template=template,
            remote_group=self.remote_group)
        return self.obj

    def _card_count(self) -> int:
        """The declared card spec: the explicit list's length, else the count."""
        ids = self.params.get('gpu_ids') or []
        return len(ids) if ids else int(self.params.get('gpu_count') or 1)


class RolloutComponent(ResourceComponent):
    """The multi-turn episode engine: a native MultiTurnRollout binding.

    Tools arrive as a wired ToolManager (shared by every episode) and/or
    through the wired ENV, whose tools the rollout binds to each episode's
    session."""

    type = 'rollout'

    IMPORTS = ('twinkle_agentic.rollout.multi_turn', 'twinkle.data_format.sampling')

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
        sampler = inputs.get('sampler')
        if sampler is None:
            raise ComponentError(f'{self.node_id}: rollout needs a wired sampler')
        template = build_template_from(
            str(self.params.get('tokenizer') or ''),
            dict(self.params.get('template_config') or {}), self.node_id,
            model_kv=dict(self.params.get('model_config') or {}))
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
                sampler, template=template, env=inputs.get('env'), tool_manager=inputs.get('tools'),
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

    IMPORTS = ('twinkle.advantage.group_admission',)

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


class _UserFormulaLoss(Loss):
    """A user formula wrapped in the native Loss interface.

    The formula is called with everything the native call site passes: the
    encoded batch (``inputs``), the model output (``outputs``) and the named
    loss kwargs (the arguments port's slots). A formula that declares fewer
    names fails loudly at the first call -- the signature is the contract,
    nothing is guessed or dropped. Channel defaults stay those of the base
    class (logps); a formula needing rarer channels belongs in a plugin that
    subclasses Loss directly.
    """

    def __init__(self, formula: Any):
        self._formula = formula

    def __call__(self, inputs: Any, outputs: Any, **lanes: Any) -> Any:
        return self._formula(inputs=inputs, outputs=outputs, **lanes)


class LossComponent(ResourceComponent):
    """L(theta) as a resident object: a registered native loss, or a formula.

    ``kind`` names the twinkle loss registry and constructs the native class
    once with typed kwargs; ``code`` replaces it with the user formula
    wrapped in the native Loss interface. The object rides the LOSS channel
    to train_step, which installs it on the model's own loss registry --
    construction here, behaviour native."""

    type = 'loss'

    IMPORTS = ('twinkle.loss',)

    def build(self, inputs: Dict[str, Any]) -> Any:
        code = str(self.params.get('code', '') or '')
        if code:
            self.obj = _UserFormulaLoss(load_code_component(code, 'loss'))
            return self.obj
        from twinkle.loss import torch_loss_mapping
        kind = str(self.params.get('kind') or 'cross_entropy')
        cls = torch_loss_mapping.get(kind)
        if cls is None:
            raise ComponentError(
                f'{self.node_id}: unknown loss {kind!r}; available kinds: '
                f'{sorted(torch_loss_mapping)}')
        kwargs = dict(self.params.get('loss_kwargs') or {})
        # Unknown names are refused BEFORE the constructor: the GRPO family
        # swallows ``**kwargs`` silently, and a panel typo must not run on
        # defaults without a trace. Type mismatches still surface from the
        # constructor itself.
        _reject_unknown_kwargs(cls, kwargs, self.node_id, f'loss {kind!r}')
        try:
            self.obj = cls(**kwargs)
        except TypeError as exc:
            raise ComponentError(
                f'{self.node_id}: loss {kind!r} rejects kwargs {sorted(kwargs)}: {exc}') from exc
        return self.obj


class ScorerComponent(ResourceComponent):
    """r(tau) as a resident callable: a registered reward family, or user code.

    ``kind`` names the ``orms`` plugin registry under the SAME dual
    construction the recipes use (SwiftPlugin rules take a filled
    RewardConfig; native rules take typed kwargs directly); ``code`` replaces
    it with the user's ``score(rows)`` callable. The object rides the SCORER
    channel to score, which calls it once per batch -- construction here,
    behaviour native."""

    type = 'scorer'

    IMPORTS = ('swift.dev.rewards',)

    def build(self, inputs: Dict[str, Any]) -> Any:
        code = str(self.params.get('code', '') or '')
        if code:
            self.obj = load_code_component(code, 'score')
            return self.obj
        from swift.dev.rewards import orms
        from swift.dev.utils.plugin import SwiftPlugin
        kind = str(self.params.get('kind') or 'math')
        cls = orms.get(kind)
        if cls is None:
            raise ComponentError(
                f'{self.node_id}: unknown scorer {kind!r}; available kinds: '
                f'{sorted(orms)} -- the registry is open, plugins extend it '
                'through swift.dev.rewards')
        kwargs = dict(self.params.get('scorer_kwargs') or {})
        # 注册表双构造协议：SwiftPlugin 型规则吃 args=Config（超参从 Config
        # 字段读，字段名校验由 _fill 承担）；twinkle 原生规则类型化旋钮直接
        # 透传，未知名先经白名单拒绝。
        if issubclass(cls, SwiftPlugin):
            self.obj = cls(args=_fill(_configs().RewardConfig, kwargs, self.node_id))
        else:
            _reject_unknown_kwargs(cls, kwargs, self.node_id, f'scorer {kind!r}')
            self.obj = cls(**kwargs)
        return self.obj


class ToolsetComponent(ResourceComponent):
    """The real tool plugins: module sources loaded through twinkle's loader.

    A plugin module's module-level Tool instances ARE the payload -- the
    declaration and the implementation are the same thing, exactly like the
    plugin path the recipes use. Runtime binding to an environment lease is
    the consumer's business (rollout / challenger)."""

    type = 'toolset'

    IMPORTS = ('twinkle.utils', 'twinkle_agentic.tools.tool_manager')

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle.utils import load_module
        from twinkle_agentic.tools.tool_manager import ToolManager
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
    """The tool-call sandbox; ``kind`` picks the native backend.

    Every backend (LocalEnv / AgentEnv / OpenEnv / OpenEnvClient) implements
    the standard Env interface, so the consumers -- the rollout env lease,
    the challenger's envs -- are kind-agnostic. ``sandbox_config`` keys must
    be the chosen backend's real constructor fields: unknown names are
    refused before construction (these constructors take ``**kwargs`` and
    would otherwise silently drop a typo and run on defaults). Credentials
    never ride the graph: agentenv's api key is read at build time from the
    environment variable named by ``api_key_env``.
    """

    type = 'sandbox_env'

    IMPORTS = ('twinkle_agentic.envs',)

    #: Constructor fields that take a Python callable. No canvas spelling can
    #: author one, so setting them here is refused with a pointer at native
    #: code rather than deferred to a confusing failure inside a rollout.
    _CALLABLE_ONLY_FIELDS = ('pre_tool_call', 'action_mapper')

    _BACKENDS = ('local', 'agentenv', 'openenv', 'openenv_client')

    def build(self, inputs: Dict[str, Any]) -> Any:
        from twinkle_agentic.envs import AgentEnv, LocalEnv, OpenEnv, OpenEnvClient
        kind = str(self.params.get('kind', 'local') or 'local')
        kv = dict(self.params.get('sandbox_config') or {})
        callables = [key for key in self._CALLABLE_ONLY_FIELDS if key in kv]
        if callables:
            raise ComponentError(
                f'{self.node_id}: sandbox_config keys {callables} take Python callables a '
                'canvas cannot author; set them through a native plugin or code component')
        cls = {'local': LocalEnv, 'agentenv': AgentEnv,
               'openenv': OpenEnv, 'openenv_client': OpenEnvClient}.get(kind)
        if cls is None:
            raise ComponentError(
                f'{self.node_id}: unknown sandbox kind {kind!r} '
                f'(available kinds: {list(self._BACKENDS)})')
        if kind == 'agentenv':
            if 'api_key' in kv:
                raise ComponentError(
                    f'{self.node_id}: credentials never ride the graph -- name the '
                    'environment variable in api_key_env instead of passing api_key')
            api_key_env = str(self.params.get('api_key_env') or '')
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if not api_key:
                    raise ComponentError(
                        f'{self.node_id}: api key env var {api_key_env!r} is unset; '
                        'export it before building the sandbox')
                kv['api_key'] = api_key
        elif self.params.get('api_key_env'):
            logger.warning('%s: api_key_env only applies to kind=agentenv; '
                           'kind=%r ignores it', self.node_id, kind)
        _reject_unknown_kwargs(cls, kv, self.node_id, f'sandbox {kind!r}')
        self.obj = cls(**kv)
        return self.obj


class ApiEndpointComponent(ResourceComponent):
    """An external OpenAI-compatible connection."""

    type = 'api_endpoint'

    IMPORTS = ('twinkle_agentic.protocol.openai',)

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
        self.obj = load_code_component(code, self.params.get('entry', 'fn'))
        return self.obj

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        """Stage form: the entry is called once per batch with the wired
        ports as keyword arguments; it returns a dict keyed by its DECLARED
        output ports.

        Stage nodes do not ride the resource pre-build channel, so the code
        loads lazily on the first invoke. A return that misses a declared
        output port is a contract violation and fails loudly -- a silently
        absent lane reads as "the round produced nothing" downstream."""
        if self.obj is None:
            self.build({})
        outputs = self.obj(**inputs)
        if not isinstance(outputs, dict):
            raise ComponentError(
                f'{self.node_id}: the code entry must return a dict keyed by its '
                'declared output ports')
        declared = [str(port.get('key') or '') for port in (self.params.get('ports_out') or [])]
        missing = [key for key in declared if key and key not in outputs]
        if missing:
            raise ComponentError(
                f'{self.node_id}: the code entry did not produce the declared '
                f'output ports {missing}')
        return outputs
