# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect122 gate: trust_remote_code configs must cross a cross-process pickle boundary.

A ray actor deserializes its creation args in a process that never imported the checkpoint's
``transformers_modules`` package; a fresh python subprocess with the same site-packages is that
exact condition (ray pickles actor args with pickle5). The real Qwen2.5-Math-PRM-7B checkpoint
(config + custom code only, no weights touched) supplies the dynamic config class.

PLAN (case | input | expected | failure it kills):
  T1 native passthrough      | LlamaConfig                       | class untouched, no marker,
                               normal pickle                     | over-patching native configs
  T2 boundary rebuild        | PRM config -> fresh child         | loads, same class, to_dict equal
                                                                   | defect122 itself
  T3 late mutation           | mutate AFTER install, then pickle | child sees the late values
                                                                   | reduce capturing state at
                                                                   install time instead of pickle time
  T4 deepcopy in-process     | deepcopy with from_pretrained     | equal+independent copy, no disk
                               poisoned                          | round-trip | __reduce__ hijacking
                                                                   deepcopy / shared-state copy
  T5 idempotent install      | make_... twice                    | one marker, pickle still faithful
                                                                   | double-wrap / marker skip dropping
                                                                   per-instance attrs
  T6 chained hop             | child rebuilds, re-pickles ->     | grandchild loads | one-hop-only fix
                               grandchild                        | (actor-side class left unpatched)
  T7 name_or_path fallback   | patched-class instance WITHOUT    | rebuilds from _name_or_path
                               instance attrs                    | closure-dir capture / missing-attr
                                                                   crash
  T8 build_config seam       | Qwen2_5PRMLoader.build_config     | dynamic class carries the marker
                                                                   | seam not wired
  T9 resolve_model_info seam | resolve_model_info(PRM)           | info.config survives the child
                                                                   | second seam not wired
"""
import copy
import json
import os
import pickle
import subprocess
import sys

import pytest
from transformers import AutoConfig, LlamaConfig

from swift.dev.utils.hf_config import _REMOTE_CODE_REDUCE_MARKER, make_remote_code_config_picklable

PRM_DIR = '/mnt/workspace/.cache/modelscope/hub/models/Qwen--Qwen2.5-Math-PRM-7B/snapshots/master'
pytestmark = pytest.mark.skipif(not os.path.isdir(PRM_DIR), reason='PRM checkpoint not on this host')

# child = the ray-actor condition: fresh interpreter, transformers_modules never imported.
_CHILD = r'''
import json, pickle, sys
assert 'transformers_modules' not in sys.modules, 'child polluted'
obj = pickle.load(open(sys.argv[1], 'rb'))
if len(sys.argv) > 2:  # chained hop: re-pickle the rebuilt object for a grandchild
    pickle.dump(obj, open(sys.argv[2], 'wb'), protocol=5)
print(json.dumps({'cls': type(obj).__name__, 'num_labels': getattr(obj, 'num_labels', None),
                  'dict': obj.to_dict()}, sort_keys=True, default=str))
'''


def _child_load(payload_path, repickle_path=None):
    argv = [sys.executable, '-c', _CHILD, payload_path]
    if repickle_path:
        argv.append(repickle_path)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f'child failed:\n{r.stderr[-3000:]}'
    return json.loads(r.stdout.strip().splitlines()[-1])


def _dump(tmp_path, name, obj):
    path = str(tmp_path / f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=5)
    return path


def _prm_config():
    return AutoConfig.from_pretrained(PRM_DIR, trust_remote_code=True)


def _json_dict(config):
    return json.loads(json.dumps(config.to_dict(), sort_keys=True, default=str))


def test_t1_native_config_passes_through_untouched(tmp_path):
    config = LlamaConfig(hidden_size=64)
    before = type(config).__dict__.copy()
    out = make_remote_code_config_picklable(config, str(tmp_path))
    assert out is config
    assert not getattr(type(config), _REMOTE_CODE_REDUCE_MARKER, False)
    assert '__reduce__' not in type(config).__dict__ or type(config).__dict__.get('__reduce__') is before.get(
        '__reduce__')
    rebuilt = pickle.loads(pickle.dumps(config, protocol=5))
    assert type(rebuilt) is LlamaConfig
    assert rebuilt.hidden_size == 64


def test_t2_remote_code_config_rebuilds_in_a_fresh_process(tmp_path):
    config = _prm_config()
    assert type(config).__module__.startswith('transformers_modules')
    make_remote_code_config_picklable(config, PRM_DIR)
    out = _child_load(_dump(tmp_path, 't2', config))
    assert out['cls'] == type(config).__name__
    assert out['dict'] == _json_dict(config)


def test_t3_mutations_after_install_are_captured_at_pickle_time(tmp_path):
    config = _prm_config()
    make_remote_code_config_picklable(config, PRM_DIR)
    config.num_labels = 7
    config.problem_type = 'regression'
    config._swift_late_attr = 'late-value'
    out = _child_load(_dump(tmp_path, 't3', config))
    # transformers 5.x num_labels is a derived property (len(id2label)), not a to_dict key: the
    # mutation travels via id2label and must re-derive on the rebuilt config.
    assert out['num_labels'] == 7
    assert len(out['dict']['id2label']) == 7
    assert out['dict']['problem_type'] == 'regression'
    assert out['dict']['_swift_late_attr'] == 'late-value'


def test_t4_deepcopy_stays_in_process_and_independent(monkeypatch):
    config = _prm_config()
    make_remote_code_config_picklable(config, PRM_DIR)

    def _no_disk(*a, **k):
        raise AssertionError('deepcopy must not round-trip through from_pretrained')

    monkeypatch.setattr(AutoConfig, 'from_pretrained', _no_disk)
    dup = copy.deepcopy(config)
    assert type(dup) is type(config)
    assert dup.to_dict() == config.to_dict()
    dup.num_labels = 99
    assert getattr(config, 'num_labels', None) != 99


def test_t5_double_install_is_idempotent(tmp_path):
    config = _prm_config()
    make_remote_code_config_picklable(config, PRM_DIR)
    first_reduce = type(config).__reduce__
    config2 = _prm_config()
    make_remote_code_config_picklable(config2, PRM_DIR)
    assert type(config2).__reduce__ is first_reduce
    assert getattr(type(config), _REMOTE_CODE_REDUCE_MARKER, False)
    out = _child_load(_dump(tmp_path, 't5', config2))
    assert out['dict'] == _json_dict(config2)


def test_t6_rebuilt_config_survives_a_second_hop(tmp_path):
    config = _prm_config()
    make_remote_code_config_picklable(config, PRM_DIR)
    hop2 = str(tmp_path / 't6_hop2.pkl')
    out1 = _child_load(_dump(tmp_path, 't6', config), repickle_path=hop2)
    out2 = _child_load(hop2)  # grandchild: the actor-side class must have been re-patched
    assert out2['cls'] == out1['cls']
    assert out2['dict'] == out1['dict']


def test_t7_falls_back_to_name_or_path_without_instance_attrs(tmp_path):
    config = _prm_config()
    make_remote_code_config_picklable(config, PRM_DIR)  # patches the class
    other = _prm_config()  # same class, but no per-instance attrs set
    assert not hasattr(other, '_swift_remote_code_dir')
    assert other._name_or_path == PRM_DIR
    out = _child_load(_dump(tmp_path, 't7', other))
    assert out['dict'] == _json_dict(other)


def test_t8_build_config_seam_installs_the_reduction():
    from swift.dev.model.loader import resolve_loader_cls, resolve_model_info

    info = resolve_model_info(PRM_DIR, task_type='seq_cls', num_labels=1)
    loader_cls = resolve_loader_cls(info.model_type, PRM_DIR)
    assert loader_cls is not None
    config = loader_cls(info).build_config(PRM_DIR)
    assert type(config).__module__.startswith('transformers_modules')
    assert getattr(type(config), _REMOTE_CODE_REDUCE_MARKER, False)


def test_t9_resolve_model_info_config_crosses_the_boundary(tmp_path):
    from swift.dev.model.loader import resolve_model_info

    info = resolve_model_info(PRM_DIR, task_type='seq_cls', num_labels=1)
    assert type(info.config).__module__.startswith('transformers_modules')
    out = _child_load(_dump(tmp_path, 't9', info.config))
    assert out['dict'] == _json_dict(info.config)