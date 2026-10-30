# Copyright (c) ModelScope Contributors. All rights reserved.
"""Execute a canvas graph declaration against the real recipes.

File-in / file-out: the canvas serialises its nodes, edges and params into a
JSON declaration; this module compiles it (:func:`compile_graph`), maps the
nodes onto the dev Config surface and drives the matching recipe. The run dir
holds the declaration copy, the stdout log and the ``summary.json`` the UI
reads back.

Regimes (detected from the GRAPH SHAPE, never from algorithm names):
offline (dataloader + offline_trainer, no rollout) runs the supervised recipes;
batch_rollout marks colocate and streaming_driver marks disaggregated -- both
route through the RL family (run_grpo / run_ppo / run_rft / run_distill), the
sampler_mode knob picking the time-sharing placement.

Algorithm identity is likewise structural: a wired value_model IS PPO, a
teacher_signal IS distillation, a trajectory_filter IS RFT, the rest is GRPO.

Single-node runs (``--target``) take the upstream closure and execute one node
with a sample batch, writing ``artifacts/<node_id>.json``. The first batch
covers the cheap data-plane nodes; heavier ones fail loudly until their
executor lands.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adapter import EdgeSpec, NodeSpec, compile_graph

# Preference-family losses route to run_dpo (rl_config.rl_type dispatch);
# everything else is a supervised objective for run_sft.
PREFERENCE_LOSSES = ('dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm')


def _info(msg: str) -> None:
    print(f'[workflow] {msg}', flush=True)


# --- declaration loading ----------------------------------------------------


def _param_str(v: Any) -> str:
    """Normalise a JSON-native param value to the canvas string convention.

    The wire contract is name -> string (the vite channel String()-ifies on the
    way in); a declaration written by hand may carry native JSON types, and the
    coercion layer downstream only splits strings -- so lists join like JS
    toString, booleans go lowercase, and null empties.
    """
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return ','.join(_param_str(x) for x in v)
    return str(v)


def load_declaration(path: Path) -> Tuple[List[NodeSpec], List[EdgeSpec], Dict[str, str]]:
    """Read the canvas JSON into compiler input plus the per-node code blobs.

    The canvas sends ``params`` as a plain dict (param name -> string value);
    ``code`` (code components) rides alongside and is keyed by node id.
    """
    raw = json.loads(path.read_text(encoding='utf-8'))
    nodes, code_blobs = [], {}
    for n in raw.get('nodes', []):
        params = {str(k): _param_str(v) for k, v in dict(n.get('params') or {}).items()}
        nodes.append(NodeSpec(id=n['id'], type=n['type'], params=params))
        if n.get('code'):
            code_blobs[n['id']] = n['code']
    edges = [EdgeSpec(e['source'], e['source_port'], e['target'], e['target_port'])
             for e in raw.get('edges', [])]
    return nodes, edges, code_blobs


# --- regime detection -------------------------------------------------------


def detect_regime(nodes: List[NodeSpec]) -> str:
    """Map the graph shape to an execution regime; structure only, no algorithm names."""
    types = {n.type for n in nodes}
    if 'streaming_driver' in types:
        return 'streaming'
    if 'batch_rollout' in types:
        return 'colocate'
    return 'offline'


# --- KV -> Config coercion ----------------------------------------------------


def _coerce(value: str, hint: Any) -> Any:
    """Coerce one string param to its field's type; unknown types pass through as str."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:  # Optional[X] -> X
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = args[0] if args else str
        origin = typing.get_origin(hint)
    if hint is bool:
        if value.lower() not in ('true', 'false'):
            raise ValueError(f'expected true/false, got {value!r}')
        return value.lower() == 'true'
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    if origin in (list, List) or hint is List or hint is list:
        parts = [p for chunk in value.split(',') for p in chunk.split() if p]
        inner = typing.get_args(hint)
        inner_hint = inner[0] if inner else str
        return [_coerce(p, inner_hint) for p in parts]
    return value  # str / Literal / Any: as-is


def fill_config(cls: Any, kv: Dict[str, str]) -> Tuple[Any, Dict[str, str]]:
    """Instantiate a Config dataclass from string params keyed by field name.

    Returns (config, unconsumed). A param whose name is not a field is NOT
    dropped silently -- the caller reports it.
    """
    hints = typing.get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    known, unconsumed = {}, {}
    for key, value in kv.items():
        if key in fields:
            if value == '' and typing.get_origin(hints[key]) is typing.Union:
                continue  # empty string means "unset" for Optional fields
            known[key] = _coerce(value, hints[key])
        else:
            unconsumed[key] = value
    return cls(**known), unconsumed


# --- supervised config assembly (shared by the offline and RL regimes) ----------


class OfflineConfigs:
    """Assembles the Config object set the supervised recipes consume.

    Node params whose names already match Config fields flow straight through
    (the registry param names are the field names); the few combined canvas
    spellings (``mini_micro_batch``) are split here. Unconsumed params are
    reported, never dropped.
    """

    def __init__(self, nodes: List[NodeSpec], edges: Optional[List[EdgeSpec]] = None):
        from swift.dev.config import (
            CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig,
            ModelConfig, RLConfig, TemplateConfig, TrainConfig, TunerConfig,
        )
        self._classes = dict(
            model=ModelConfig, dataset=DatasetConfig, train=TrainConfig,
            checkpoint=CheckpointConfig, logging=LoggingConfig, distributed=DistributedConfig,
            template=TemplateConfig, tuner=TunerConfig, rl=RLConfig,
        )
        self._nodes = nodes
        self._edges = edges or []
        self.unconsumed: Dict[str, Dict[str, str]] = {}
        self._consumed_keys: set = set()
        self._loss: Optional[str] = None

    def _node_kv(self, *types: str) -> Dict[str, str]:
        kv: Dict[str, str] = {}
        for n in self._nodes:
            if n.type in types:
                kv.update(n.params)
        return kv

    def _fill(self, key: str, kv: Dict[str, str]) -> Any:
        config, left = fill_config(self._classes[key], kv)
        self._consumed_keys.update(kv.keys() - left.keys())
        return config

    def build(self) -> Dict[str, Any]:
        """Split the combined canvas spellings, then fill every Config class.

        The policy node carries the training-face params (loss / optim / lr) alongside
        the model ones; they are routed to TrainConfig/CheckpointConfig by FIELD OWNERSHIP
        (a param belongs to whichever Config declares its name), not by hand-listed maps.
        """
        configs = self._supervised_configs()
        loss = self._loss
        if loss in PREFERENCE_LOSSES:
            kv = {'rl_type': loss}
            # The preference reference (frozen copy wired to the trainer's ref port).
            refs, _, _ = _reference_roles(self._nodes, self._edges)
            if refs:
                kv['ref_model'] = refs[0]
                self._consumed_keys.add('model')
            configs['rl_config'] = self._fill('rl', kv)
        self._report_unconsumed()
        return configs

    def _supervised_configs(self) -> Dict[str, Any]:
        """The Config set every supervised recipe consumes, before the regime face.

        Both the offline regime (run_sft/run_dpo) and the RL family (run_grpo and
        siblings) start from this same assembly; the RL subclass layers the rollout /
        rl / reward faces on top.
        """
        merged = self._node_kv('policy_model')
        # The template-face params ride on the policy node too (encoding, truncation,
        # loss mask): peel them off BEFORE the model/tuner split, or ModelConfig --
        # which knows none of these names -- would strand them as unconsumed.
        template_kv = {k: v for k, v in merged.items() if k in _TEMPLATE_FIELDS}
        rest = {k: v for k, v in merged.items() if k not in _TEMPLATE_FIELDS}
        tuner_kv = {k: v for k, v in rest.items() if k in _TUNER_FIELDS}
        model_kv = {k: v for k, v in rest.items() if k not in _TUNER_FIELDS}
        # train_type is the canvas spelling of TunerConfig.tuner (a tier, not a fork).
        if 'train_type' in tuner_kv:
            tuner_kv['tuner'] = tuner_kv.pop('train_type')
            self._consumed_keys.add('train_type')  # canonical canvas name, rewritten above

        train_kv = self._node_kv('offline_trainer', 'train_step')
        ckpt_kv = dict(train_kv)
        if 'mini_micro_batch' in train_kv:
            mini, _, micro = train_kv.pop('mini_micro_batch').partition('/')
            train_kv.setdefault('per_device_train_batch_size', mini.strip())
            train_kv.setdefault('gradient_accumulation_steps', micro.strip())
        ckpt_kv.pop('mini_micro_batch', None)
        # The dataloader node's knobs live on the dataset/train configs under different names.
        dataset_kv = self._node_kv('dataset')
        for key, value in self._node_kv('dataloader').items():
            config_key, field = _LOADER_RENAMES.get(key, (None, None))
            if config_key == 'train' and field not in train_kv:
                train_kv[field] = value
            elif config_key == 'dataset' and field not in dataset_kv:
                dataset_kv[field] = value
            self._consumed_keys.add(key)
        # parallel_spec rides the policy node (placement declaration); the empty
        # string means "single-device default" for the Optional field.
        dist_kv: Dict[str, str] = {'parallel_spec': merged.get('parallel_spec', '')}
        # Route the policy node's training-face params by Config field ownership.
        train_fields = {f.name for f in dataclasses.fields(self._classes['train'])}
        ckpt_fields = {f.name for f in dataclasses.fields(self._classes['checkpoint'])}
        dist_fields = {f.name for f in dataclasses.fields(self._classes['distributed'])}
        for key, value in merged.items():
            if key in train_fields and key not in train_kv:
                train_kv[key] = value
            elif key in ckpt_fields and key not in ckpt_kv:
                ckpt_kv[key] = value
            elif key in dist_fields and key not in dist_kv:
                dist_kv[key] = value
        # Single-process spawn is the physical fact of the executor: the CLI gets
        # nproc from torchrun / WORLD_SIZE, here it must be pinned explicitly.
        dist_kv.setdefault('nproc_per_node', '1')
        self._dist_kv = dict(dist_kv)

        configs: Dict[str, Any] = dict(
            model_config=self._fill('model', model_kv),
            template_config=self._fill('template', template_kv),
            dataset_config=self._fill('dataset', dataset_kv),
            train_config=self._fill('train', train_kv),
            distributed_config=self._fill('distributed', dist_kv),
            checkpoint_config=self._fill('checkpoint', ckpt_kv),
            logging_config=self._fill('logging', {}),
            tuner_config=self._fill('tuner', tuner_kv),
        )
        from swift.dev.cli.parser import select_tuner
        configs['tuner_config'] = select_tuner(configs['tuner_config'])
        self._loss = train_kv.get('loss')
        return configs

    def _report_unconsumed(self) -> None:
        """A param consumed by ANY config is consumed; only true orphans get reported."""
        consumed = set(self._consumed_keys)
        consumed.update({'parallel_spec', 'mini_micro_batch'})  # split / routed spellings
        orphans: Dict[str, Dict[str, str]] = {}
        for n in self._nodes:
            for key in n.params:
                if key not in consumed:
                    orphans.setdefault(n.id, {})[key] = n.params[key]
        self.unconsumed = orphans

    @property
    def loss(self) -> str:
        return self._loss


# Fields that belong to TunerConfig, not ModelConfig (split from the policy node).
_TUNER_FIELDS = {'train_type', 'tuner', 'adapters', 'lora_rank', 'lora_alpha', 'lora_dropout',
                 'use_rslora', 'use_dora', 'target_modules', 'modules_to_save'}

# Template-face fields (TemplateConfig names): the encoding/truncation/loss-mask
# knobs the canvas exposes on the policy node. Empty string means "leave default".
_TEMPLATE_FIELDS = {'template', 'max_length', 'truncation_strategy', 'loss_scale', 'system',
                    'padding_side', 'padding_free', 'enable_thinking', 'response_prefix'}

# The dataloader node's canvas spellings -> (config, real field name).
_LOADER_RENAMES = {
    'batch_size': ('train', 'per_device_train_batch_size'),
    'num_workers': ('dataset', 'dataloader_num_workers'),
    'shuffle': ('dataset', 'train_dataloader_shuffle'),
    'drop_last': ('dataset', 'dataloader_drop_last'),
}


# --- RL regime: rollout / rl / reward faces on the shared assembly --------------


#: The distillation family: one recipe, the loss param picks the signal form.
DISTILL_LOSSES = ('gkd', 'opsd', 'mopd')

#: Node params whose canvas spelling differs from the RL Config field name.
#: Keyed by (node type, param name) -> (config key, real field name).
_RL_RENAMES = {
    ('advantage_estimator', 'estimator'): ('rl', 'advantage_estimator'),
    ('advantage_estimator', 'scale'): ('rl', 'scale_rewards'),
    ('advantage_estimator', 'num_generations'): ('rl', 'num_generations'),
    ('reward_scorer', 'scorer'): ('reward', 'orm'),
    ('reward_scorer', 'weights'): ('reward', 'orm_weights'),
    ('sampler', 'engine'): ('rollout', 'rollout_sampler'),
    ('sampler', 'max_model_len'): ('rollout', 'vllm_max_model_len'),
    ('sampler', 'gpu_memory_util'): ('rollout', 'vllm_gpu_memory_utilization'),
    ('sampler', 'sleep_level'): ('rollout', 'sleep_level'),
    ('streaming_driver', 'async_mode'): ('rollout', 'async_mode'),
    ('streaming_driver', 'max_staleness'): ('rollout', 'max_staleness'),
    ('streaming_driver', 'num_generations'): ('rl', 'num_generations'),
    ('batch_rollout', 'num_generations'): ('rl', 'num_generations'),
    ('batch_rollout', 'temperature'): ('generation', 'temperature'),
    ('batch_rollout', 'max_new_tokens'): ('generation', 'max_new_tokens'),
    ('weight_sync', 'strategy'): ('rollout', 'weight_sync_strategy'),
    ('weight_sync', 'backend'): ('rollout', 'weight_sync_backend'),
    ('teacher_signal', 'teacher_weights'): ('rl', 'teacher_weights'),
    ('teacher_signal', 'gkd_logits_topk'): ('rl', 'gkd_logits_topk'),
    ('trajectory_filter', 'select'): ('rl', 'rft_select'),
    ('trajectory_filter', 'threshold'): ('rl', 'rft_threshold'),
    ('trajectory_filter', 'top_k'): ('rl', 'rft_top_k'),
    ('multi_turn_rollout', 'max_turns'): ('rollout', 'max_turns'),
    ('train_step', 'publish_cadence'): ('rollout', 'parameter_sync_step'),
}

#: Where a frozen_model's wiring sends its weights: the role is the EDGE.
#: ref -> KL/preference anchor, teacher -> distillation teacher (many),
#: scorer -> a scoring model id in the reward channel.
_REF_ROUTES = {
    ('advantage_estimator', 'ref'): 'ref',
    ('offline_trainer', 'ref'): 'ref',
    ('train_step', 'ref'): 'ref',
    ('teacher_signal', 'teacher'): 'teacher',
    ('reward_scorer', 'model'): 'scorer',
    ('streaming_driver', 'score'): 'scorer',
}


def _reference_roles(nodes: List[NodeSpec],
                     edges: List[EdgeSpec]) -> Tuple[List[str], List[str], List[str]]:
    """Split the wired frozen models by role: (refs, teachers, scorers).

    A same-weights placeholder ("同策略...") means the recipe should reuse the
    policy's own checkpoint -- it contributes nothing here.
    """
    by_id = {n.id: n for n in nodes}
    refs, teachers, scorers = [], [], []
    for e in edges:
        source, target = by_id.get(e.source), by_id.get(e.target)
        if source is None or target is None or source.type != 'frozen_model':
            continue
        model = source.params.get('model', '')
        if not model or model.startswith('同策略'):
            continue
        role = _REF_ROUTES.get((target.type, e.target_port))
        if role == 'ref':
            refs.append(model)
        elif role == 'teacher':
            teachers.append(model)
        elif role == 'scorer':
            scorers.append(model)
    return refs, teachers, scorers


class RLConfigs(OfflineConfigs):
    """The RL family's config assembly: the supervised face plus rollout / rl /
    reward / generation.

    Reference models route BY WIRING -- the same frozen copy is a KL anchor on
    ``advantage.ref``, a teacher on ``teacher_signal.teacher`` or a scoring model
    on ``driver.score``; which one it is lives in the edge, not the node.
    """

    def __init__(self, nodes: List[NodeSpec], edges: List[EdgeSpec]):
        super().__init__(nodes, edges)
        from swift.dev.config import GenerationConfig, RewardConfig, RolloutConfig
        self._classes.update(rollout=RolloutConfig, reward=RewardConfig,
                             generation=GenerationConfig)
        self._edges = edges

    def _rl_type(self) -> str:
        """Algorithm identity from graph shape; no algorithm names on the canvas.

        A wired value model IS PPO, a teacher signal IS distillation (the loss
        param picks gkd/opsd/mopd), a trajectory filter IS RFT; the rest is GRPO.
        """
        types = {n.type for n in self._nodes}
        if 'value_model' in types:
            return 'ppo'
        if 'teacher_signal' in types:
            loss = self._loss or 'gkd'
            return loss if loss in DISTILL_LOSSES else 'gkd'
        if 'trajectory_filter' in types:
            return 'rft'
        return 'grpo'

    def build(self) -> Dict[str, Any]:
        configs = self._supervised_configs()
        rl_kv: Dict[str, str] = {'rl_type': self._rl_type()}
        # The regime picks the sampler placement: batch_rollout time-shares the
        # trainer devices (colocate), streaming_driver keeps a separate engine.
        types = {n.type for n in self._nodes}
        rollout_kv: Dict[str, str] = {
            'sampler_mode': 'colocate' if 'batch_rollout' in types else 'disaggregated'}
        reward_kv: Dict[str, str] = {}
        gen_kv: Dict[str, str] = {}
        for n in self._nodes:
            for key, value in n.params.items():
                cfg, field = _RL_RENAMES.get((n.type, key), (None, None))
                if cfg is None:
                    continue
                self._consumed_keys.add(key)
                {'rl': rl_kv, 'rollout': rollout_kv,
                 'reward': reward_kv, 'generation': gen_kv}[cfg].setdefault(field, value)
        # Reference models: one frozen copy, as many roles as it has edges.
        refs, teachers, scorers = _reference_roles(self._nodes, self._edges)
        if refs:
            rl_kv.setdefault('ref_model', refs[0])
            self._consumed_keys.add('model')
            # EMA shadow: only a ref-WIRED frozen node's ema flag turns on the
            # sync (rl.sync_ref_model); an ema flag on a teacher/scorer wiring is
            # meaningless and must not silently flip the RL face.
            by_id = {n.id: n for n in self._nodes}
            for e in self._edges:
                source, target = by_id.get(e.source), by_id.get(e.target)
                if (source is not None and target is not None and source.type == 'frozen_model'
                        and _REF_ROUTES.get((target.type, e.target_port)) == 'ref'
                        and source.params.get('ema', 'false').lower() == 'true'):
                    rl_kv['sync_ref_model'] = 'true'
                    self._consumed_keys.add('ema')
        if teachers:
            rl_kv.setdefault('teacher_model', ','.join(teachers))
            self._consumed_keys.add('model')
        if scorers:
            reward_kv.setdefault('orm', ','.join(scorers))
            self._consumed_keys.add('model')
        # The canvas generation knob feeds BOTH faces: generation.max_new_tokens (reports) and
        # rl.max_completion_length (what the rollout engine actually generates). Without the
        # second route the rollout silently generates RLConfig's 512-token default.
        if 'max_new_tokens' in gen_kv:
            rl_kv.setdefault('max_completion_length', gen_kv['max_new_tokens'])
            self._consumed_keys.add('max_new_tokens')
        # generation_batch_size bounds the per-window rollout batch. Left unset it falls back to
        # len(prompts) -- the WHOLE dataset in one window -- which multiplies every request by
        # num_generations and the completion budget: a graph that never finishes its first batch.
        train_cfg = configs.get('train_config')
        if train_cfg is not None and 'generation_batch_size' not in rollout_kv:
            per_batch = getattr(train_cfg, 'per_device_train_batch_size', None) or 1
            ga = getattr(train_cfg, 'gradient_accumulation_steps', None) or 1
            rollout_kv['generation_batch_size'] = str(int(per_batch) * int(ga))
        configs['rollout_config'] = self._fill('rollout', rollout_kv)
        # Online RL is Ray-mode by construction: the trainer and the rollout
        # sampler are separate actors the driver syncs weights between.
        rl_dist = dict(getattr(self, '_dist_kv', {}))
        rl_dist.setdefault('mode', 'ray')
        configs['distributed_config'] = self._fill('distributed', rl_dist)
        configs['rl_config'] = self._fill('rl', rl_kv)
        if reward_kv:
            configs['reward_config'] = self._fill('reward', reward_kv)
        if gen_kv:
            configs['generation_config'] = self._fill('generation', gen_kv)
        self._report_unconsumed()
        return configs


def run_rl(nodes: List[NodeSpec], edges: List[EdgeSpec], run_dir: Path,
           regime: str) -> Dict[str, Any]:
    """Colocate / streaming regime: one RL recipe call, the loop lives inside it."""
    builder = RLConfigs(nodes, edges)
    configs = builder.build()
    if builder.unconsumed:
        for key, left in builder.unconsumed.items():
            _info(f'warn: unconsumed {key} params: {sorted(left)}')
    for n in nodes:
        if n.type not in ('dataset', 'dataloader', 'policy_model', 'offline_trainer',
                          'train_step', 'sampler', 'streaming_driver', 'batch_rollout',
                          'advantage_estimator', 'reward_scorer', 'weight_sync',
                          'teacher_signal', 'trajectory_filter', 'value_model',
                          'frozen_model', 'multi_turn_rollout'):
            _info(f'note: node {n.id} ({n.type}) is a post-training node -- single-run it separately')

    rl_type = configs['rl_config'].rl_type
    if rl_type in DISTILL_LOSSES:
        from swift.dev.recipe.distill import run_distill as run
        _info(f'recipe: run_distill (rl_type={rl_type})')
    elif rl_type == 'ppo':
        from swift.dev.recipe.ppo import run_ppo as run
        _info('recipe: run_ppo')
    elif rl_type == 'rft':
        from swift.dev.recipe.rft import run_rft as run
        _info('recipe: run_rft')
    else:
        from swift.dev.recipe.grpo import run_grpo as run
        _info('recipe: run_grpo')
    output_dir = str(run_dir / 'train')
    history = run(output_dir=output_dir, **configs)
    _info(f'history rows: {len(history)}')
    return {'regime': regime, 'rl_type': rl_type, 'output_dir': output_dir,
            'history': history[-5:]}


def run_offline(nodes: List[NodeSpec], edges: List[EdgeSpec], run_dir: Path) -> Dict[str, Any]:
    """Offline regime: one recipe call, the loop lives inside it."""
    builder = OfflineConfigs(nodes, edges)
    configs = builder.build()
    if builder.unconsumed:
        for key, left in builder.unconsumed.items():
            _info(f'warn: unconsumed {key} params: {sorted(left)}')
    for n in nodes:
        if n.type not in ('dataset', 'dataloader', 'policy_model', 'offline_trainer', 'train_step'):
            _info(f'note: node {n.id} ({n.type}) is a post-training node -- single-run it separately')

    if 'rl_config' in configs:  # preference family
        from swift.dev.recipe.rlhf import run_dpo as run
        _info('recipe: run_dpo (preference family)')
    else:
        from swift.dev.recipe.sft import run_sft as run
        _info('recipe: run_sft (supervised)')
    output_dir = str(run_dir / 'train')
    history = run(output_dir=output_dir, **configs)
    _info(f'history rows: {len(history)}')
    return {'regime': 'offline', 'output_dir': output_dir, 'history': history[-5:]}


# --- single-node runs (cheap data-plane and construction-only nodes) ------------


def _find_upstream_ckpt(run_dir: Path) -> Optional[str]:
    """The post-training chain's upstream: a finished training checkpoint.

    This run's own ``train/`` first (single-run right after run-all), then the
    most recent historical run directories -- the chain nodes are one-shot
    actions over whatever the training segment last produced.
    """
    own = run_dir / 'train' / 'checkpoint-final'
    if own.is_dir():
        return str(own)
    try:
        siblings = sorted(run_dir.parent.iterdir(), key=lambda p: p.stat().st_mtime,
                          reverse=True)
    except OSError:
        return None
    for run_root in siblings[:8]:
        ckpt = run_root / 'train' / 'checkpoint-final'
        if ckpt.is_dir():
            return str(ckpt)
    return None


def _configured_plugin_sources() -> List[str]:
    """External plugin sources registered on the plugins page (~/.swift/plugins.json).

    The webui registers each plugin as one ``--external_plugins`` source; the single-node
    toolset run loads them all so ``@register('tool', ...)`` calls take effect.
    """
    path = Path.home() / '.swift' / 'plugins.json'
    try:
        entries = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    return [e['source'] for e in entries if isinstance(e, dict) and e.get('source')]


def run_single(target: str, nodes: List[NodeSpec], steps: List[Any], run_dir: Path,
               code_blobs: Optional[Dict[str, str]] = None,
               edges: Optional[List[EdgeSpec]] = None) -> int:
    """Execute one node with its upstream closure; heavy node types fail loudly for now."""
    edges = edges or []
    by_id = {n.id: n for n in nodes}
    node = by_id.get(target)
    if node is None:
        _info(f'error: target node {target!r} not in the graph')
        return 2
    supported = {'dataset', 'dataloader', 'api_model', 'sandbox_env', 'code_component',
                 'toolset',
                 'train_step', 'offline_trainer',
                 'merge', 'infer', 'deploy', 'quantize', 'hub_push'}
    if node.type not in supported:
        _info(f'error: single-run for {node.type} lands with the later batches; '
              f'use run-all for the training segment')
        return 2

    artifacts = run_dir / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    if node.type == 'api_model':
        from twinkle_agentic.protocol.openai import OpenAI
        base_url = node.params.get('base_url', '') or None
        model = node.params.get('model', '') or 'unknown'
        # Local compatible endpoints (vLLM / SGLang / twinkle-server) take any
        # placeholder key; the SDK refuses to construct without one.
        client = OpenAI(model=model, base_url=base_url, api_key='EMPTY')  # no request sent
        summary = {'kind': 'MODEL', 'model': model, 'base_url': base_url or '(sdk default)',
                   'protocol': 'openai-compatible chat.completions'}
        del client
    elif node.type == 'sandbox_env':
        from twinkle_agentic.envs import LocalEnv
        env = LocalEnv()  # kind='local'; pooled / microVM kinds arrive with their batch
        summary = {'kind': 'ENV', 'env': type(env).__name__,
                   'workspace': getattr(env, 'workspace_root', None) or 'temp dir per lease'}
        del env
    elif node.type == 'code_component':
        code = (code_blobs or {}).get(node.id, '')
        if not code:
            _info('error: this code component has no function body on the canvas')
            return 2
        # Drive the user transform on a two-row stub batch: single-run is about
        # seeing the OUTPUT SHAPE, not the data (the real batch rides run-all).
        stub = [{'messages': [{'role': 'user', 'content': '2+2=?'}]},
                {'messages': [{'role': 'user', 'content': '3+3=?'}]}]
        namespace: Dict[str, Any] = {}
        exec(code, namespace)  # noqa: S102 -- the canvas code IS the payload
        fn = namespace.get('fn')
        if fn is None:  # fall back to the LAST callable defined in the blob
            fn = next((v for v in reversed(list(namespace.values())) if callable(v)), None)
        if fn is None:
            _info('error: the code defines no callable (name it fn, or make the last def the entry)')
            return 2
        out = fn(stub)
        rows = out if isinstance(out, list) else [out]
        summary = {'kind': 'TRAJ', 'rows': len(rows), 'head': rows[:2],
                   'entry': getattr(fn, '__name__', '<lambda>')}
    elif node.type == 'toolset':
        # One-shot tool assembly: load the registered plugin sources, resolve the named
        # tool plugins, and build a ToolManager -- the same surface a rollout would get.
        from swift.dev.builders.sandbox import resolve_tool_plugins
        from swift.dev.utils.plugin import PluginRegistry
        from twinkle_agentic.tools.tool_manager import ToolManager
        names = [t.strip() for t in node.params.get('tools', '').split(',') if t.strip()]
        if not names:
            _info('error: toolset needs registered tool plugin names in its tools param')
            return 2
        sources = _configured_plugin_sources()
        if sources:
            _info(f'loading plugin sources: {sources}')
            PluginRegistry.load_external(sources)
        plugins = resolve_tool_plugins(names, None)
        tools = [tool for plugin in plugins for tool in plugin.build(None)]
        if not tools:
            _info(f'error: tool plugins {names} built no tools')
            return 2
        manager = ToolManager(tools)
        # ToolInfo is a TypedDict (OpenAI shape: type + function{name, description, parameters})
        schemas = [dict(info) for info in manager.tool_infos()]
        summary = {'kind': 'TOOL', 'tools': [s['function']['name'] for s in schemas],
                   'schemas': schemas}
    elif node.type in ('train_step', 'offline_trainer'):
        # A batch through one round = ONE optimizer step (the whole-run path with
        # max_steps pinned to 1); the model load is the known minute-scale cost.
        if detect_regime(nodes) != 'offline':
            _info('error: RL train-step single-run needs the full RL assembly -- use run-all')
            return 2
        builder = OfflineConfigs(nodes, edges)
        configs = builder.build()
        configs['train_config'] = dataclasses.replace(configs['train_config'], max_steps=1)
        if configs.get('rl_config') is not None:  # preference family
            from swift.dev.recipe.rlhf import run_dpo as run
            _info('recipe: run_dpo (one round)')
        else:
            from swift.dev.recipe.sft import run_sft as run
            _info('recipe: run_sft (one round)')
        history = run(output_dir=str(run_dir / 'train'), **configs)
        row = history[0] if history else {}
        ckpt = run_dir / 'train' / 'checkpoint-final'
        summary = {'kind': 'CKPT', 'steps_done': len(history), 'last_step': row,
                   'ckpt': str(ckpt) if ckpt.is_dir() else None}
    elif node.type in ('merge', 'infer', 'deploy', 'quantize', 'hub_push'):
        # Post-training chain: one-shot actions over the training products.
        # merge/infer consume the ADAPTER checkpoint (base model + adapters);
        # deploy/quantize/hub_push need the MERGED full weights.
        merged = run_dir / 'merged'
        ckpt = _find_upstream_ckpt(run_dir)
        supervised = OfflineConfigs(nodes, edges)._supervised_configs()
        model_config = supervised['model_config']  # the BASE model stays
        template_config = supervised['template_config']
        if node.type in ('merge', 'infer'):
            if ckpt is None:
                _info('error: no upstream checkpoint -- run the training segment first')
                return 2
            _info(f'upstream adapter: {ckpt}')
        elif not merged.is_dir():
            _info('error: no merged weights in this run -- single-run the merge node first '
                  '(deploy/quantize/hub_push consume full weights)')
            return 2
        else:
            _info(f'upstream merged weights: {merged}')
            model_config = dataclasses.replace(model_config, model=str(merged))
        if node.type == 'merge':
            if node.params.get('merge_lora', 'true').lower() == 'false':
                # Pure export (the declared form of merge for full-finetune ckpts):
                # the checkpoint already IS full weights.
                output = ckpt
                _info('note: merge_lora=false -> pure export, the checkpoint is the artifact')
            else:
                from swift.dev.recipe.export import run_merge_lora
                tuner_config = dataclasses.replace(supervised['tuner_config'],
                                                   adapters=[ckpt])
                output = run_merge_lora(model_config, tuner_config,
                                        template_config=template_config,
                                        output_dir=str(merged))
            summary = {'kind': 'MODEL', 'path': output}
        elif node.type == 'infer':
            from swift.dev.config import DatasetConfig, InferConfig
            from swift.dev.recipe.infer import run_infer
            infer_config, _ = fill_config(
                InferConfig, {'result_path': str(run_dir / 'predictions.jsonl'),
                              **node.params})
            dataset_kv = {k: v for k, v in node.params.items()
                          if k in {f.name for f in dataclasses.fields(DatasetConfig)}}
            dataset_config, _ = fill_config(DatasetConfig, dataset_kv)
            run_infer(model_config, template_config, dataset_config, infer_config,
                      adapters=[ckpt])
            rows = 0
            result = run_dir / 'predictions.jsonl'
            if result.exists():
                rows = sum(1 for _ in result.open(encoding='utf-8'))
            summary = {'kind': 'DATA', 'rows': rows, 'result': str(result)}
        elif node.type == 'deploy':
            from swift.dev.config import DeployConfig
            from swift.dev.recipe.infer import run_deploy
            deploy_kv = dict(node.params)
            if 'infer_backend' in deploy_kv:  # canvas spelling -> DeployConfig field
                deploy_kv['sampler_type'] = deploy_kv.pop('infer_backend')
            deploy_kv.setdefault('detach', 'true')  # single-run must return, not serve forever
            deploy_config, _ = fill_config(DeployConfig, deploy_kv)
            _info('deploy: detached deployment starting (endpoint registration in models.json)')
            run_deploy(model_config, template_config, deploy_config=deploy_config)
            summary = {'kind': 'ENDPOINT', 'model': str(merged),
                       'served_model_name': deploy_config.served_model_name or str(merged)}
        elif node.type == 'quantize':
            from swift.dev.config import QuantizeConfig
            from swift.dev.recipe.export import run_quantize
            quantize_config, _ = fill_config(QuantizeConfig, node.params)
            output = run_quantize(model_config, template_config, quantize_config,
                                  output_dir=str(run_dir / 'quantized'))
            summary = {'kind': 'MODEL', 'path': output}
        else:  # hub_push
            from swift.dev.config import CheckpointConfig, DatasetConfig
            from swift.dev.recipe.export import run_push_to_hub
            if not node.params.get('hub_model_id'):
                _info('error: hub_push needs hub_model_id (the target repo)')
                return 2
            checkpoint_config, _ = fill_config(CheckpointConfig, node.params)
            dataset_config, _ = fill_config(DatasetConfig, {})
            url = run_push_to_hub(str(merged), checkpoint_config, dataset_config,
                                  node.params.get('commit_message', 'webui workflow push'))
            summary = {'kind': 'MODEL', 'path': str(merged), 'hub_url': url}
    elif node.type == 'dataset':
        from swift.dev.recipe.grpo.run_grpo import prompt_rows_from_dataset
        from swift.dev.config import DatasetConfig
        config, _ = fill_config(DatasetConfig, node.params)
        rows, prompts, _ = prompt_rows_from_dataset(config)
        summary = {'kind': 'DATA', 'rows': len(prompts), 'head': prompts[:2]}
    else:  # dataloader
        from swift.dev.builders import build_prompt_dataloader
        from swift.dev.recipe.grpo.run_grpo import prompt_rows_from_dataset
        from swift.dev.config import DatasetConfig
        dataset_node = next((n for n in nodes if n.type == 'dataset'), None)
        if dataset_node is None:
            _info('error: single-run the dataloader needs a dataset node upstream')
            return 2
        config, _ = fill_config(DatasetConfig, dataset_node.params)
        rows, prompts, extras = prompt_rows_from_dataset(config)
        loader = build_prompt_dataloader(prompts, extras, seed=42, batch_size=8)
        batch = next(iter(loader))
        summary = {'kind': 'DATA', 'batch': len(batch), 'head': batch[:2]}
    (artifacts / f'{node.id}.json').write_text(
        json.dumps({'node': node.id, 'type': node.type, 'status': 'done', 'output': summary},
                   ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    _info(f'node {node.id} done -> artifacts/{node.id}.json')
    return 0


# --- entry ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    graph_path = Path(argv[0])
    target = None
    run_dir = graph_path.parent
    it = iter(argv[1:])
    for arg in it:
        if arg == '--target':
            target = next(it)
        elif arg == '--run-dir':
            run_dir = Path(next(it))
    run_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges, code_blobs = load_declaration(graph_path)
    steps = compile_graph(nodes, edges)  # validates wiring, orders construction
    _info(f'compiled {len(steps)} nodes, regime={detect_regime(nodes)}')
    if target:
        return run_single(target, nodes, steps, run_dir, code_blobs, edges)

    regime = detect_regime(nodes)
    try:
        if regime == 'offline':
            result = run_offline(nodes, edges, run_dir)
        else:
            result = run_rl(nodes, edges, run_dir, regime)
    except Exception as exc:  # the task executor surfaces this via the log
        _info(f'error: {exc!r}')
        raise
    status = {'status': 'done', 'target': None, **result,
              'nodes': [{'id': n.id, 'type': n.type, 'status': 'done'} for n in nodes]}
    (run_dir / 'summary.json').write_text(json.dumps(status, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    _info('done -> summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
