# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weightless model-structure and trainable-parameter inspection.

The engine behind the webui structure panel; ``swift.dev.webui.inspection`` adds the stdio
protocol. No training path imports this module.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import multiprocessing
import os
import re
import struct
import subprocess
import tempfile
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from swift.dev.utils.hub import safe_snapshot_download

SCHEMA_VERSION = 1
_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class StructureNode:
    id: str
    parent_id: Optional[str]
    name: str
    module_path: str
    kind: str
    class_name: Optional[str] = None
    part: str = 'other'
    repeat: Optional[int] = None
    params: Optional[int] = None
    children: List[str] = field(default_factory=list)
    source: str = 'config'
    module_paths: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    training: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelStructure:
    detail: Literal['config', 'meta']
    model: Dict[str, Any]
    config_fingerprint: str
    nodes: List[StructureNode]
    summary: Dict[str, Any]
    requested_detail: Literal['config', 'meta'] = 'config'
    warnings: List[Dict[str, str]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        value = asdict(self)
        value['nodes'] = [node.to_dict() for node in self.nodes]
        return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()


def _config_dict(config: Any) -> dict:
    if isinstance(config, dict):
        return config
    return config.to_dict()


def _sub_config(config: dict, names: tuple[str, ...]) -> dict:
    for name in names:
        value = config.get(name)
        if isinstance(value, dict):
            return value
    return config


def _known_values(config: dict, names: tuple[str, ...]) -> dict:
    return {name: config[name] for name in names if isinstance(config.get(name), (int, float, str))}


def _layer_count(config: dict) -> Optional[int]:
    for name in ('num_hidden_layers', 'num_layers', 'n_layer', 'num_transformer_layers', 'depth'):
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _safetensors_param_count(model_dir: str) -> Optional[int]:
    paths = list(Path(model_dir).glob('*.safetensors'))
    if not paths:
        return None
    total = 0
    try:
        for path in paths:
            with path.open('rb') as file:
                header_size = struct.unpack('<Q', file.read(8))[0]
                metadata = json.loads(file.read(header_size))
            for name, tensor in metadata.items():
                if name != '__metadata__':
                    total += math.prod(tensor.get('shape', []))
    except (OSError, ValueError, struct.error, json.JSONDecodeError):
        return None
    return total or None


def _config_param_estimate(config: dict) -> Optional[int]:
    hidden = config.get('hidden_size')
    layers = _layer_count(config)
    vocab = config.get('vocab_size')
    intermediate = config.get('intermediate_size')
    if not all(isinstance(value, int) and value > 0 for value in (hidden, layers, vocab, intermediate)):
        return None
    experts = config.get('num_experts') or config.get('moe_num_experts') or 1
    return vocab * hidden + layers * (12 * hidden * hidden + 3 * hidden * intermediate * experts)


@dataclass
class _ConfigInfo:
    model_dir: str
    model_name: str
    model_type: Optional[str]
    config: dict
    architectures: List[str]
    torch_dtype: str
    is_multimodal: bool
    is_moe_model: bool
    tags: List[str]


#: Multimodal partition per family, mirroring ``ModelArch`` in ``loader/*.py`` (cross-checked by
#: ``test_config_arch_matches_loader_arch``); copied rather than resolved so config mode imports
#: no training stack. A family absent here renders as a plain LLM; the meta view is authoritative.
_CONFIG_ARCH: Tuple[Tuple[Tuple[str, ...], Dict[str, List[str]]], ...] = (
    (('qwen3vl', ), {
        'language_model': ['model.language_model', 'lm_head'],
        'vision_tower': ['model.visual'],
        'aligner': ['model.visual.merger', 'model.visual.deepstack_merger_list'],
    }),
    (('qwen35', ), {
        'language_model': ['model.language_model', 'lm_head'],
        'vision_tower': ['model.visual'],
        'aligner': ['model.visual.merger'],
    }),
    (('qwen25vl', 'qwen2vl'), {
        'language_model': ['model.language_model', 'lm_head'],
        'vision_tower': ['model.visual'],
        'aligner': ['model.visual.merger'],
    }),
    (('qwen3omni', ), {
        'language_model': ['thinker.model', 'thinker.lm_head'],
        'vision_tower': ['thinker.audio_tower', 'thinker.visual'],
        'aligner': [
            'thinker.audio_tower.proj1', 'thinker.audio_tower.proj2', 'thinker.visual.merger',
            'thinker.visual.merger_list'
        ],
        'generator': ['talker', 'code2wav'],
    }),
    (('qwen25omni', ), {
        'language_model': ['thinker.model', 'thinker.lm_head'],
        'vision_tower': ['thinker.audio_tower', 'thinker.visual'],
        'aligner': ['thinker.audio_tower.proj', 'thinker.visual.merger'],
        'generator': ['talker', 'token2wav'],
    }),
    (('idefics3', ), {
        'language_model': ['model.text_model'],
        'vision_tower': ['model.vision_model'],
        'aligner': ['model.connector'],
    }),
    (('gemma3n', ), {
        'language_model': ['model.language_model', 'lm_head'],
        'vision_tower': ['model.vision_tower', 'model.audio_tower'],
        'aligner': ['model.embed_vision', 'model.embed_audio'],
    }),
    (('llavaonevision15', ), {
        'language_model': ['model.language_model', 'lm_head'],
        'vision_tower': ['model.visual'],
        'aligner': ['model.visual.merger'],
    }),
    (('gemma3', 'pixtral', 'mistral3'), {
        'language_model': ['model.language_model', 'lm_head'],
        'vision_tower': ['model.vision_tower'],
        'aligner': ['model.multi_modal_projector'],
    }),
    (('llava', ), {
        'language_model': ['model.language_model', 'lm_head'],
        'vision_tower': ['model.vision_tower'],
        'aligner': ['model.multi_modal_projector'],
    }),
    (('llama4', ), {
        'language_model': ['language_model'],
        'vision_tower': ['vision_model'],
        'aligner': ['multi_modal_projector'],
    }),
)


def _config_arch(model_type: Optional[str], architectures: List[str]) -> Dict[str, List[str]]:
    key = ''.join(char for char in ' '.join([model_type or '', *architectures]).lower() if char.isalnum())
    for markers, arch in _CONFIG_ARCH:
        if any(marker in key for marker in markers):
            return arch
    return {}


def _cached_model_dir(model: str, revision: Optional[str]) -> Optional[str]:
    """Probe local hub layouts first: config mode must resolve offline."""
    model_id, _, subfolder = model.partition(':')
    repo = model_id.strip('/').replace('/', '--')
    revision = revision or 'master'
    roots = []
    ms_cache = os.environ.get('MODELSCOPE_CACHE')
    if ms_cache:
        roots.append(Path(ms_cache) / 'models')
    roots.extend((Path.home() / '.cache/modelscope/hub/models', Path.home() / '.cache/modelscope/models'))
    for root in roots:
        snapshot = root / repo / 'snapshots' / revision
        if (snapshot / 'config.json').is_file():
            return str(snapshot / subfolder if subfolder else snapshot)
    hf_repo = 'models--' + repo
    hf_root = os.environ.get('HF_HUB_CACHE') or os.path.join(
        os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface'), 'hub')
    base = Path(hf_root) / hf_repo
    snapshot = base / 'snapshots' / ('main' if revision == 'master' else revision)
    ref = base / 'refs' / ('main' if revision == 'master' else revision)
    if not snapshot.is_dir() and ref.is_file():
        snapshot = base / 'snapshots' / ref.read_text(encoding='utf-8').strip()
    if (snapshot / 'config.json').is_file():
        return str(snapshot / subfolder if subfolder else snapshot)
    return None


def _resolve(model_config):
    source = os.path.expanduser(model_config.model)
    local_path = '/'.join(source.split(':', 1))
    if os.path.exists(source) or os.path.exists(local_path):
        model_dir = os.path.abspath(source if os.path.exists(source) else local_path)
    else:
        model_dir = _cached_model_dir(model_config.model, model_config.model_revision)
        if model_dir is None:
            model_dir = safe_snapshot_download(
                model_config.model, revision=model_config.model_revision, download_model=False)
    with open(Path(model_dir) / 'config.json', encoding='utf-8') as file:
        raw_config = json.load(file)
    architectures = list(raw_config.get('architectures') or [])
    model_type = model_config.model_type or raw_config.get('model_type')
    is_multimodal = any(key in raw_config for key in (
        'vision_config', 'visual_config', 'vit_config', 'audio_config', 'thinker_config'))
    text_config = _sub_config(raw_config, ('text_config', 'language_config', 'llm_config'))
    is_moe = 'moe' in ''.join(architectures).lower() or any(key in text_config for key in (
        'num_experts', 'num_experts_per_tok', 'n_routed_experts', 'moe_intermediate_size'))
    info = _ConfigInfo(
        model_dir=model_dir,
        model_name=model_config.model.rstrip('/').rsplit('/', 1)[-1].replace('___', '.'),
        model_type=model_type,
        config=raw_config,
        architectures=architectures,
        torch_dtype=str(raw_config.get('torch_dtype') or raw_config.get('dtype') or ''),
        is_multimodal=is_multimodal,
        is_moe_model=is_moe,
        tags=['multimodal'] if is_multimodal else [],
    )
    # A text-only checkpoint of a multimodal family (e.g. gemma3_text) has no towers to split.
    return info, (_config_arch(model_type, architectures) if is_multimodal else {})


def _fingerprint(model_config, config: Any) -> str:
    try:
        swift_version = version('ms-swift')
    except PackageNotFoundError:
        swift_version = 'unknown'
    try:
        transformers_version = version('transformers')
    except PackageNotFoundError:
        transformers_version = 'unknown'
    payload = {
        'model': model_config.model,
        'revision': model_config.model_revision,
        'config': _config_dict(config),
        'model_config': asdict(model_config),
        'transformers': transformers_version,
        'swift': swift_version,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _cache_key(fingerprint: str, detail: str, expand: Optional[str], training: bool, tuner_config,
               train_config, frozen: bool, backend: str, parallel: dict) -> str:
    payload = {
        'fingerprint': fingerprint,
        'detail': detail,
        'expand': expand,
        'training': training,
        'tuner': asdict(tuner_config) if tuner_config is not None else None,
        'train': asdict(train_config) if train_config is not None else None,
        'frozen': frozen,
        'backend': backend,
        'parallel': parallel,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    with _CACHE_LOCK:
        value = _CACHE.get(key)
        return copy.deepcopy(value) if value is not None else None


def _cache_put(key: str, value: dict) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) >= 32:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = copy.deepcopy(value)


def _estimated(status: str) -> dict:
    return {'status': status, 'source': 'config', 'verified': False}


def _normalize_backend(value: Any) -> str:
    return 'megatron' if str(value or '').lower() == 'megatron' else 'hf'


def _normalize_parallel(value: Any) -> dict:
    """Dependency-free subset of ``DeviceMesh.from_spec``: config mode imports no twinkle."""
    if isinstance(value, str):
        value = dict(re.findall(r'([a-z]+)(\d+)', value.lower()))
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ('dp', 'fsdp', 'tp', 'pp', 'cp', 'ep', 'etp', 'epfsdp', 'vpp', 'ul'):
        try:
            number = int(value.get(key) or 1)
        except (TypeError, ValueError):
            continue
        if number > 1:
            result[key] = number
    if value.get('sp') is True:
        result['sp'] = True
    return result


def _is_full(tuner_config) -> bool:
    return tuner_config is None or tuner_config.tuner == 'full'


def _has_full_freeze_rules(train_config) -> bool:
    if train_config is None:
        return False
    return bool(train_config.freeze_parameters_ratio or train_config.freeze_parameters
                or train_config.freeze_parameters_regex or train_config.trainable_parameters
                or train_config.trainable_parameters_regex)


def _apply_router_training(nodes: List[StructureNode], tuner_config, frozen: bool, backend: str) -> None:
    if backend != 'megatron' or frozen or _is_full(tuner_config):
        return
    router = next((node for node in nodes if node.id == 'llm.decoder.router'), None)
    if router is not None and tuner_config.tuner == 'lora':
        router.training = _estimated('frozen_base')


def _apply_config_training(nodes: List[StructureNode], tuner_config, train_config, frozen: bool,
                           backend: str = 'hf') -> None:
    if frozen:
        for node in nodes:
            node.training = _estimated('frozen_base')
        return
    if _is_full(tuner_config):
        has_rules = _has_full_freeze_rules(train_config)
        for node in nodes:
            status = 'unknown' if has_rules and node.parent_id is not None else (
                'mixed' if has_rules else 'full_trainable')
            node.training = _estimated(status)
        return
    if tuner_config.tuner != 'lora':
        for node in nodes:
            node.training = _estimated('unknown' if node.parent_id is not None else 'mixed')
        return
    all_linear = not tuner_config.target_regex and 'all-linear' in (tuner_config.target_modules or [])
    frozen_parts = {
        'llm': tuner_config.freeze_llm,
        'vit': tuner_config.freeze_vit,
        'aligner': tuner_config.freeze_aligner,
        'generator': True,
    }
    for node in nodes:
        if node.parent_id is None:
            status = 'mixed'
        elif all_linear and node.part in frozen_parts:
            status = 'frozen_base' if frozen_parts[node.part] else 'adapter_estimated'
        else:
            status = 'unknown'
        node.training = _estimated(status)
    _apply_router_training(nodes, tuner_config, frozen, backend)


def _config_structure(model_config,
                      *,
                      training: bool,
                      tuner_config,
                      train_config,
                      frozen: bool,
                      backend: str = 'hf',
                      parallel: Optional[dict] = None) -> dict:
    parallel = _normalize_parallel(parallel)
    info, arch = _resolve(model_config)
    raw_config = _config_dict(info.config)
    text_config = _sub_config(raw_config, ('text_config', 'language_config', 'llm_config'))
    vision_config = _sub_config(raw_config, ('vision_config', 'visual_config', 'vit_config'))
    fingerprint = _fingerprint(model_config, info.config)
    total_params = _safetensors_param_count(info.model_dir)
    total_params_source = 'safetensors' if total_params is not None else None
    if total_params is None:
        total_params = _config_param_estimate(text_config)
        total_params_source = 'config-estimate' if total_params is not None else None
    model = {
        'source': model_config.model,
        'name': info.model_name,
        'model_dir': info.model_dir,
        'model_type': info.model_type,
        'revision': model_config.model_revision,
        'torch_dtype': str(info.torch_dtype).removeprefix('torch.'),
        'architectures': info.architectures,
        'is_multimodal': info.is_multimodal,
        'is_moe': info.is_moe_model,
        'tags': info.tags,
        'backend': backend,
        'parallel': parallel,
        'arch': arch,
        'config': {
            'text': _known_values(text_config, (
                'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                'num_experts', 'n_routed_experts', 'num_experts_per_tok', 'vocab_size')),
            'vision': _known_values(vision_config, (
                'hidden_size', 'intermediate_size', 'num_heads', 'patch_size', 'depth')),
        },
    }
    nodes: List[StructureNode] = []

    def add(node_id: str, parent_id: Optional[str], name: str, kind: str, *, part: str = 'other',
            repeat: Optional[int] = None, details: Optional[dict] = None) -> StructureNode:
        node = StructureNode(node_id, parent_id, name, '', kind, part=part, repeat=repeat,
                             details=details or {})
        nodes.append(node)
        if parent_id is not None:
            next(parent for parent in nodes if parent.id == parent_id).children.append(node_id)
        return node

    add('root', None, info.model_name, 'model', details={'architectures': info.architectures})
    if arch.get('vision_tower') or info.is_multimodal:
        vit = add('vit', 'root', 'Vision / Audio Tower', 'tower', part='vit', repeat=_layer_count(vision_config),
                  details=model['config']['vision'])
        vit.module_paths = list(arch.get('vision_tower') or [])
        vit.module_path = vit.module_paths[0] if vit.module_paths else ''
    if arch.get('aligner') or info.is_multimodal:
        aligner = add('aligner', 'root', 'Aligner / Projector', 'aligner', part='aligner')
        aligner.module_paths = list(arch.get('aligner') or [])
        aligner.module_path = aligner.module_paths[0] if aligner.module_paths else ''
    if arch.get('generator'):
        generator = add('generator', 'root', 'Generator', 'generator', part='generator')
        generator.module_paths = list(arch['generator'])
        generator.module_path = generator.module_paths[0]
    llm = add('llm', 'root', 'Language Model', 'tower', part='llm', details=model['config']['text'])
    llm.module_paths = list(arch.get('language_model') or [])
    llm.module_path = llm.module_paths[0] if llm.module_paths else ''
    add('llm.embedding', 'llm', 'Token Embedding', 'embedding', part='llm')
    add('llm.decoder', 'llm', 'Decoder Layer', 'repeated', part='llm', repeat=_layer_count(text_config),
        details=model['config']['text'])
    add('llm.decoder.attention', 'llm.decoder', 'Attention', 'attention', part='llm')
    mlp_name = 'MoE / MLP' if info.is_moe_model else 'MLP'
    add('llm.decoder.mlp', 'llm.decoder', mlp_name, 'moe' if info.is_moe_model else 'mlp', part='llm')
    if backend == 'megatron' and info.is_moe_model:
        add('llm.decoder.router', 'llm.decoder', 'MoE Router', 'router', part='llm')
    add('llm.norm', 'llm', 'Final Norm', 'norm', part='llm')
    add('llm.head', 'llm', 'LM Head', 'head', part='llm')
    if training:
        _apply_config_training(nodes, tuner_config, train_config, frozen, backend)
    training_summary = nodes[0].training if training else None
    structure = ModelStructure(
        'config', model, fingerprint, nodes, {
            'total_params': total_params,
            'total_params_source': total_params_source,
            'training': training_summary,
        })
    return structure.to_dict()


def _base_module_path(path: str) -> str:
    prefix = 'base_model.model.'
    while path.startswith(prefix):
        path = path[len(prefix):]
    if path == 'base_model.model':
        return ''
    return path


def _part_for_path(path: str, arch) -> str:
    if not any((arch.language_model, arch.vision_tower, arch.aligner, arch.generator)):
        return 'llm'
    path = _base_module_path(path)
    candidates = []
    for part, prefixes, priority in (('aligner', arch.aligner, 0), ('generator', arch.generator, 1),
                                     ('vit', arch.vision_tower, 2), ('llm', arch.language_model, 3)):
        for prefix in prefixes:
            if path == prefix or path.startswith(f'{prefix}.'):
                candidates.append((-len(prefix), priority, part))
    return min(candidates)[2] if candidates else 'other'


def _parameter_status(path: str, requires_grad: bool, tuner_config) -> str:
    segments = path.split('.')
    if 'modules_to_save' in segments:
        return 'trainable_copy' if requires_grad else 'frozen_copy'
    if any(segment.startswith('lora_') for segment in segments):
        return 'adapter_trainable' if requires_grad else 'frozen_adapter'
    if requires_grad:
        return 'full_trainable' if _is_full(tuner_config) else 'trainable_base'
    return 'frozen_base'


def _training_counts() -> dict:
    return {
        'total_params': 0,
        'trainable_params': 0,
        'adapter_params': 0,
        'modules_to_save_params': 0,
        'frozen_params': 0,
    }


def _direct_training(path: str, module, tuner_config) -> dict:
    counts = _training_counts()
    statuses = set()
    parameters = []
    for name, parameter in module.named_parameters(recurse=False):
        full_path = f'{path}.{name}' if path else name
        size = parameter.numel()
        status = _parameter_status(full_path, parameter.requires_grad, tuner_config)
        statuses.add(status)
        counts['total_params'] += size
        counts['trainable_params'] += size if parameter.requires_grad else 0
        counts['adapter_params'] += size if status in ('adapter_trainable', 'frozen_adapter') else 0
        counts['modules_to_save_params'] += size if status in ('trainable_copy', 'frozen_copy') else 0
        parameters.append({
            'name': name,
            'shape': list(parameter.shape),
            'dtype': str(parameter.dtype).removeprefix('torch.'),
            'status': status,
        })
    counts['frozen_params'] = counts['total_params'] - counts['trainable_params']
    counts.update({
        'status': next(iter(statuses)) if len(statuses) == 1 else ('mixed' if statuses else 'empty'),
        'source': 'meta',
        'verified': True,
        'parameters': parameters,
    })
    return counts


def _merge_training(target: dict, source: dict) -> None:
    for key in ('total_params', 'trainable_params', 'adapter_params', 'modules_to_save_params', 'frozen_params'):
        target[key] += source[key]


def _training_status(counts: dict, tuner_config) -> str:
    trainable = counts['trainable_params']
    if not counts['total_params']:
        return 'empty'
    if not trainable:
        return 'frozen_base'
    if counts['adapter_params'] == trainable and not counts['modules_to_save_params']:
        return 'adapter_trainable'
    if trainable == counts['total_params'] and _is_full(tuner_config):
        return 'full_trainable'
    return 'mixed'


def _aggregate_training(nodes: Dict[str, StructureNode], tuner_config) -> dict:
    """Roll every node's direct counts up its subtree; returns the root summary."""

    def rollup(node: StructureNode) -> dict:
        total = _training_counts()
        direct = dict(node.training)
        parameters = direct.pop('parameters', [])
        _merge_training(total, direct)
        for child_id in node.children:
            _merge_training(total, rollup(nodes[child_id]))
        total.update({
            'status': _training_status(total, tuner_config),
            'source': 'meta',
            'verified': True,
            'parameters': parameters,
        })
        node.training = total
        return total

    return rollup(nodes['root'])


def _remove_nodes(nodes: Dict[str, StructureNode], node_id: str) -> None:
    for child_id in list(nodes[node_id].children):
        _remove_nodes(nodes, child_id)
    nodes.pop(node_id)


def _collapse_group(nodes: Dict[str, StructureNode], parent: StructureNode, group: List[StructureNode],
                    tuner_config) -> None:
    group.sort(key=lambda item: int(item.id.rsplit('.', 1)[-1]))
    representative = group[0]
    for sibling in group[1:]:
        if representative.training:
            _merge_training(representative.training, sibling.training)
        parent.children.remove(sibling.id)
        _remove_nodes(nodes, sibling.id)
    if representative.training:
        representative.training['status'] = _training_status(representative.training, tuner_config)
    representative.name = f'{representative.class_name} × {len(group)}'
    representative.kind = 'repeated'
    representative.repeat = len(group)
    representative.module_path = parent.module_path
    representative.module_paths = [item.id for item in group]
    representative.params = sum(item.params or 0 for item in group)
    representative.details['collapsed'] = True


def _collapse_repeated(nodes: Dict[str, StructureNode], expand: Optional[str] = None, tuner_config=None) -> None:

    def visit(node_id: str) -> None:
        node = nodes[node_id]
        for child_id in list(node.children):
            visit(child_id)
        groups: Dict[str, List[StructureNode]] = {}
        for child_id in node.children:
            if child_id.rsplit('.', 1)[-1].isdigit():
                child = nodes[child_id]
                groups.setdefault(child.class_name or '', []).append(child)
        for group in groups.values():
            if len(group) >= 2 and expand not in (node.module_path, _base_module_path(node.module_path)):
                _collapse_group(nodes, node, group, tuner_config)

    visit('root')


def _assert_meta(model) -> None:
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != 'meta':
            raise RuntimeError(f'meta inspection allocated a real tensor: {name} on {tensor.device}')


def _apply_meta_training(model, model_config, arch, tuner_config, train_config, frozen: bool):
    if _is_full(tuner_config):
        if frozen:
            model.requires_grad_(False)
        elif train_config is not None:
            from twinkle.model.base import _freeze_then_activate
            _freeze_then_activate(
                list(model.named_parameters()),
                freeze_ratio=train_config.freeze_parameters_ratio,
                freeze_names=train_config.freeze_parameters,
                freeze_regex=train_config.freeze_parameters_regex,
                trainable_names=train_config.trainable_parameters,
                trainable_regex=train_config.trainable_parameters_regex)
        return model
    from peft import get_peft_model
    from twinkle.model.transformers.target_modules import expand_modules_to_save, expand_multimodal_lora_target

    from swift.dev.builders.adapter import _build_adapter_config

    adapter_config = _build_adapter_config(tuner_config, task_type=model_config.task_type)
    expand_multimodal_lora_target(
        model,
        adapter_config,
        asdict(arch),
        freeze_llm=tuner_config.freeze_llm,
        freeze_vit=tuner_config.freeze_vit,
        freeze_aligner=tuner_config.freeze_aligner)
    expand_modules_to_save(model, adapter_config)
    model = get_peft_model(model, adapter_config)
    if frozen:
        model.requires_grad_(False)
    return model


def _meta_structure(model_config, *, expand: Optional[str] = None, training: bool, tuner_config, train_config,
                    frozen: bool, base: Optional[dict] = None) -> dict:
    # Training-stack imports stay here: config mode must neither pay nor require them.
    import torch
    from swift.dev.builders.model import load_model_processor
    from swift.dev.model.loader import ModelArch, resolve_loader_cls

    if base is None:
        base = _config_structure(
            model_config, training=training, tuner_config=tuner_config, train_config=train_config, frozen=frozen)
    loader_cls = resolve_loader_cls(model_config.model_type, model_config.model)
    arch = loader_cls(None).model_arch if loader_cls is not None else ModelArch(**base['model']['arch'])
    with torch.device('meta'):
        model, _ = load_model_processor(
            copy.deepcopy(model_config), load_model=True, return_dummy_model=True, download_model=False)
        if training:
            model = _apply_meta_training(model, model_config, arch, tuner_config, train_config, frozen)
        _assert_meta(model)
    nodes: Dict[str, StructureNode] = {}
    for path, module in model.named_modules():
        node_id = path or 'root'
        parent_id = path.rpartition('.')[0] if path else None
        if parent_id == '':
            parent_id = 'root'
        direct_params = sum(parameter.numel() for _, parameter in module.named_parameters(recurse=False))
        node = StructureNode(
            node_id,
            parent_id,
            path.rsplit('.', 1)[-1] if path else base['model']['name'],
            path,
            'adapter' if any(segment.startswith('lora_') for segment in path.split('.')) else 'module',
            class_name=module.__class__.__name__,
            part=_part_for_path(path, arch),
            params=direct_params,
            source='meta',
            training=_direct_training(path, module, tuner_config) if training else None)
        if path and parent_id not in nodes:
            raise RuntimeError(f'module parent was not visited before {path}: {parent_id}')
        nodes[node_id] = node
        if parent_id is not None:
            nodes[parent_id].children.append(node_id)
    if training:
        training_summary = _aggregate_training(nodes, tuner_config)
    else:
        training_summary = None
    _collapse_repeated(nodes, expand, tuner_config)
    ordered_nodes = list(nodes.values())
    total_params = sum(parameter.numel() for _, parameter in model.named_parameters())
    structure = ModelStructure(
        'meta',
        base['model'],
        base['config_fingerprint'],
        ordered_nodes,
        {'total_params': total_params, 'total_params_source': 'meta', 'training': training_summary},
        requested_detail='meta')
    return structure.to_dict()


def _meta_process_entry(result_path: str, log_path: str, model_config, *, expand: Optional[str], training: bool,
                        tuner_config, train_config, frozen: bool, base: dict) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with open(log_path, 'w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            result = _meta_structure(
                model_config,
                expand=expand,
                training=training,
                tuner_config=tuner_config,
                train_config=train_config,
                frozen=frozen,
                base=base)
        except Exception:
            result = {'error': traceback.format_exc()}
        with open(result_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False)
    os._exit(0)


def _run_meta_worker(model_config, tuner_config, train_config, *, expand: Optional[str], training: bool,
                     frozen: bool, timeout_s: float, base: dict) -> dict:
    # Load registries before fork so the worker's timeout only covers model construction.
    from swift.dev.model import loader as _loader
    del _loader
    model_config = copy.copy(model_config)
    model_config.torch_dtype = model_config.torch_dtype or base['model']['torch_dtype']
    # Local checkpoint dirs carry no registered id: reuse the config tier's model_type so the
    # builder resolves the family instead of failing on an unrecognized directory name.
    model_config.model_type = model_config.model_type or base['model']['model_type']
    method = 'fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'
    context = multiprocessing.get_context(method)
    with tempfile.TemporaryDirectory() as directory:
        result_path = os.path.join(directory, 'result.json')
        log_path = os.path.join(directory, 'worker.log')
        process = context.Process(
            target=_meta_process_entry,
            args=(result_path, log_path, model_config),
            kwargs={
                'expand': expand,
                'training': training,
                'tuner_config': tuner_config,
                'train_config': train_config,
                'frozen': frozen,
                'base': base,
            })
        process.start()
        process.join(timeout_s)
        if process.exitcode is None:
            process.terminate()
            process.join(5)
            if process.exitcode is None:
                process.kill()
                process.join()
            raise subprocess.TimeoutExpired('meta worker', timeout_s)
        if not os.path.exists(result_path):
            message = f'meta worker exited with code {process.exitcode}'
            if os.path.exists(log_path):
                with open(log_path, encoding='utf-8') as file:
                    message += ': ' + file.read()[-4000:]
            raise RuntimeError(message)
        with open(result_path, encoding='utf-8') as file:
            result = json.load(file)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def inspect_model(model_config,
                  *,
                  detail: Literal['config', 'meta'] = 'config',
                  expand: Optional[str] = None,
                  training: bool = False,
                  tuner_config=None,
                  train_config=None,
                  frozen: bool = False,
                  backend: str = 'hf',
                  parallel: Optional[dict] = None,
                  timeout_s: float = 60.0) -> dict:
    """Inspect a checkpoint without loading weights; meta failures retain the config view.

    ``expand`` keeps one module's subtree unfolded for callers needing per-layer paths.
    """
    if detail not in ('config', 'meta'):
        raise ValueError("detail must be 'config' or 'meta'")
    if not model_config.model:
        raise ValueError('model_config.model is required')
    if timeout_s <= 0:
        raise ValueError('timeout_s must be positive')
    backend = _normalize_backend(backend)
    parallel = _normalize_parallel(parallel)
    fallback = _config_structure(
        model_config,
        training=training,
        tuner_config=tuner_config,
        train_config=train_config,
        frozen=frozen,
        backend=backend,
        parallel=parallel)
    config_key = _cache_key(
        fallback['config_fingerprint'], 'config', expand, training, tuner_config, train_config, frozen, backend,
        parallel)
    cached = _cache_get(config_key)
    fallback = cached or fallback
    _cache_put(config_key, fallback)
    if detail == 'config':
        return fallback
    if backend == 'megatron':
        fallback['requested_detail'] = 'meta'
        fallback['warnings'].append({
            'code': 'megatron_meta_unsupported',
            'message': 'Megatron exact inspection needs a live mcore model and is not supported by the static view.',
        })
        return fallback
    meta_key = _cache_key(
        fallback['config_fingerprint'], 'meta', expand, training, tuner_config, train_config, frozen, backend,
        parallel)
    cached = _cache_get(meta_key)
    if cached is not None:
        return cached
    try:
        result = _run_meta_worker(
            model_config,
            tuner_config,
            train_config,
            expand=expand,
            training=training,
            frozen=frozen,
            timeout_s=timeout_s,
            base=fallback)
    except Exception as exc:
        fallback['requested_detail'] = 'meta'
        fallback['warnings'].append({
            'code': 'meta_timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'meta_failed',
            'message': str(exc),
        })
        return fallback
    _cache_put(meta_key, result)
    return result


def _clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
