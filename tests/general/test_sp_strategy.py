# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-only unit tests for SPStrategy guards and config semantics.

Run (from the repository root, no NPU / no torch.distributed needed):

    pytest tests/general/test_sp_strategy.py -v

Covers the contract added during the M3 review fixes:
  - enabled=False blocks initialize()
  - hooks raise RuntimeError before initialize()
  - strategy ulysses_size wins over the sp_size argument (with a warning)
  - sp_size <= 1 is a no-op
  - gather_logits=False bypasses postprocess_outputs gathering
  - runtime methods (gather/pad/pad_and_split_inputs/gather_object_dp/
    create_sp_sampler/create_sp_dispatcher) raise before initialize()
  - read-only topology properties pass through to the singleton
  - dp factories hand the sampler/dispatcher only dp topology, not the
    SP engine instance
"""
import pytest

from swift.sequence_parallel.sequence_parallel import sequence_parallel
from swift.sequence_parallel.strategy import SPConfig, SPStrategy


def test_enabled_false_blocks_initialize():
    s = SPStrategy(sp_config=SPConfig(enabled=False))
    assert s.initialize(sp_size=2) is False
    assert s._initialized is False


def test_hooks_raise_before_initialize():
    s = SPStrategy()
    with pytest.raises(RuntimeError, match='before initialize'):
        s.preprocess_inputs({})
    with pytest.raises(RuntimeError, match='before initialize'):
        s.postprocess_outputs(None, None)
    with pytest.raises(RuntimeError, match='before initialize'):
        s.gather_loss_tensors(None, None, 1)


def test_sp_size_le_1_is_noop(monkeypatch):
    monkeypatch.setattr(sequence_parallel, 'prepare',
                        lambda *a, **k: pytest.fail('prepare must not be called when sp_size <= 1'))
    s = SPStrategy()
    assert s.initialize(sp_size=1) is False
    assert s.initialize(sp_size=None) is False
    assert s._initialized is False


def test_strategy_ulysses_size_wins_with_warning(monkeypatch):
    calls = []
    warnings = []
    monkeypatch.setattr(sequence_parallel, 'prepare', lambda *a, **k: calls.append((a, k)))
    # swift's logger does not propagate to the root logger, so caplog cannot see it;
    # patch the strategy module's logger directly instead.
    from swift.sequence_parallel import strategy as strategy_mod
    monkeypatch.setattr(strategy_mod.logger, 'warning', lambda msg: warnings.append(msg))
    s = SPStrategy(sp_config=SPConfig(ulysses_size=2))
    assert s.initialize(sp_size=4, model='m', tokenizer='t') is True
    # The strategy value (2) must win over the call-site argument (4).
    assert calls[0][0][0] == 2
    assert calls[0][1]['model'] == 'm'
    assert any('differs' in w for w in warnings)


def test_gather_logits_false_bypasses_gather():
    s = SPStrategy(sp_config=SPConfig(gather_logits=False))
    s._initialized = True  # pretend initialize() succeeded; the gate must short-circuit first
    preds, labels = object(), object()
    out_preds, out_labels = s.postprocess_outputs(preds, labels)
    assert out_preds is preds
    assert out_labels is labels


def test_runtime_methods_raise_before_initialize():
    s = SPStrategy()
    calls = [
        lambda: s.gather(None, 1),
        lambda: s.pad(None, 0),
        lambda: s.pad_and_split_inputs(None, None, None, None, None, None),
        lambda: s.gather_object_dp([]),
        lambda: s.create_sp_sampler(None),
        lambda: s.create_sp_dispatcher(None),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match='before initialize'):
            call()


def test_readonly_properties_passthrough(monkeypatch):
    # real_position_ids/dp_rank/dp_group are @property on the singleton class;
    # patch at class level (or via extra_kwargs), world_size is a plain attribute.
    monkeypatch.setattr(sequence_parallel, 'extra_kwargs', {'text_position_ids': 'pos'})
    monkeypatch.setattr(type(sequence_parallel), 'dp_rank', property(lambda self: 3))
    monkeypatch.setattr(type(sequence_parallel), 'dp_group', property(lambda self: 'G'))
    monkeypatch.setattr(sequence_parallel, 'world_size', 8)
    monkeypatch.setattr(sequence_parallel, 'rp_world_size', 2)
    s = SPStrategy()
    assert s.real_position_ids == 'pos'
    assert s.dp_rank == 3
    assert s.dp_group == 'G'
    assert s.world_size == 8
    assert s.rp_world_size == 2


def test_gather_and_gather_object_dp_forward(monkeypatch):
    calls = []
    monkeypatch.setattr(sequence_parallel, 'gather', lambda *a, **k: calls.append(('gather', a, k)) or 'G')
    monkeypatch.setattr(sequence_parallel, '_gather_object_dp', lambda x: calls.append(
        ('gather_object_dp', x)) or ['X'])
    s = SPStrategy()
    s._initialized = True
    assert s.gather('t', dim=1, position_ids='p') == 'G'
    assert calls[0] == ('gather', ('t', ), {'dim': 1, 'position_ids': 'p'})
    assert s.gather_object_dp(['d']) == ['X']
    assert calls[1] == ('gather_object_dp', ['d'])


def test_pad_forwards_to_singleton(monkeypatch):
    monkeypatch.setattr(sequence_parallel, 'pad', lambda *a, **k: (a, k))
    monkeypatch.setattr(sequence_parallel, 'pad_and_split_inputs', lambda *a, **k: (a, k))
    s = SPStrategy()
    s._initialized = True
    assert s.pad('t', padding_value=-1) == (('t', ), {'padding_value': -1})
    args, kwargs = s.pad_and_split_inputs(None, None, 'labels', None, None, None, real_position_ids='p')
    assert args == (None, None, 'labels', None, None, None)
    assert kwargs == {'real_position_ids': 'p'}


def test_create_sp_sampler_narrows_to_dp_topology(monkeypatch):
    from swift.sequence_parallel import utils as sp_utils
    captured = {}

    class FakeDataDim:

        def get_group(self):
            return 'DP_GROUP'

    monkeypatch.setattr(sequence_parallel, 'device_mesh', {'data': FakeDataDim()}, raising=False)
    monkeypatch.setattr(sp_utils, 'SequenceParallelSampler', lambda *a, **k: captured.update(a=a, k=k) or 'SAMPLER')
    s = SPStrategy()
    s._initialized = True
    assert s.create_sp_sampler('DS', seed=42) == 'SAMPLER'
    assert captured['a'] == ('DS', )
    assert captured['k'] == {'dp_group': 'DP_GROUP', 'shuffle': True, 'seed': 42, 'round_up': True}


def test_create_sp_dispatcher_narrows_to_dp_topology(monkeypatch):
    from swift.sequence_parallel import utils as sp_utils
    captured = {}
    monkeypatch.setattr(type(sequence_parallel), 'dp_rank', property(lambda self: 3))
    monkeypatch.setattr(sequence_parallel, 'dp_world_size', 4)
    monkeypatch.setattr(type(sequence_parallel), 'dp_group', property(lambda self: 'G'))
    monkeypatch.setattr(sp_utils, 'SequenceParallelDispatcher',
                        lambda *a, **k: captured.update(a=a, k=k) or 'DISPATCHER')
    s = SPStrategy()
    s._initialized = True
    assert s.create_sp_dispatcher('DL', device='npu', skip_batches=2) == 'DISPATCHER'
    assert captured['a'] == ('DL', )
    assert captured['k'] == {
        'dp_rank': 3,
        'dp_world_size': 4,
        'dp_group': 'G',
        'device': 'npu',
        'skip_batches': 2,
    }
