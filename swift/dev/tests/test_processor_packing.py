"""Packing must survive the WHOLE InputProcessor pipeline, not just the dataset split.

Why this file exists: packing is a three-stage relay -- PackingDataset decides which samples share
a sequence, the InputProcessor flattens + concatenates them into one variable-length sequence, and
the model isolates attention via cu_seqlens. Each stage looked fine in isolation, yet packing was
broken end to end (a packed item is a list[dict], which the processor never flattened -> AttributeError;
and the position_ids that make packing detectable were injected for the megatron framework only).

The pre-existing tests only covered _pack's construction (mocked) and the unpack helpers in
isolation, so nothing ever pushed real packed data through the processor -- which is exactly why the
bug survived. These tests do that, and assert the packed LAYOUT (multiple 0-resets in position_ids
+ correct cu_seqlens), because packing's failure mode is silently-wrong numerics rather than a crash:
without the reset form, flash-attn treats the concatenation as ONE sequence and attention leaks
across samples while loss still looks reasonable.
"""
from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from swift.dev.processor import InputProcessor


def _packed_batch():
    """One packed item = list of rows, exactly what PackingDataset.__getitem__ yields (packing.py:130).

    Two rows of length 3 and 2; identity_collate passes the list through, so the processor receives
    [[row, row]] (a batch of one packed group).
    """
    return [[
        {
            'input_ids': [1, 2, 3],
            'labels': [1, 2, 3]
        },
        {
            'input_ids': [4, 5],
            'labels': [4, 5]
        },
    ]]


def _processor(framework: str) -> InputProcessor:
    return InputProcessor(framework=framework, padding_free=True)


@pytest.mark.parametrize('framework', ['transformers', 'megatron'])
def test_packed_batch_flattened_and_position_ids_reset(framework):
    """A packed list[list[dict]] must flatten and get per-row position_ids on BOTH backends.

    Flattening mirrors legacy Template.data_collator (template/base.py:1668-1669). The per-row
    position_ids are the producer of the packed layout: each row gets range(len(input_ids)), so the
    later concatenation yields the multiple-0-reset form. Legacy's packing_row is backend-agnostic,
    so a transformers-only or megatron-only injection would leave the other backend's packed batch
    without position_ids -- and flash-attn would then attend across sample boundaries silently.
    """
    prepared = _processor(framework).prepare_inputs(_packed_batch())

    # flattened: one packed group of 2 rows -> 2 features (not 1 list, not a crash)
    assert len(prepared) == 2, f'packed batch not flattened: {prepared}'
    # each row carries its OWN 0-based position_ids
    pos = [p['position_ids'].flatten().tolist() for p in prepared]
    assert pos == [[0, 1, 2], [0, 1]], f'per-row position_ids wrong: {pos}'


def test_packed_collate_produces_packed_position_ids_and_cu_seqlens():
    """End-to-end: the collated micro batch must be detectable as packed, with right cu_seqlens.

    This is the load-bearing assertion. After prepare_inputs + collate_fn under padding_free,
    twinkle's _collate_macro_batch concatenates the rows, so position_ids [0,1,2] + [0,1] become
    [0,1,2,0,1]. That reset pattern is what _is_packed_position_ids detects and what
    _get_packed_seq_params turns into cu_seqlens=[0,3,5] (sequence boundaries) -- the signal that
    keeps attention inside each original sample. If flattening or the position_ids injection
    regresses, position_ids degrades to a single monotonic run and this assertion fails.
    """
    processor = _processor('transformers')
    collated = processor.collate_fn(processor.prepare_inputs(_packed_batch()))
    batch = collated[0]

    assert batch['input_ids'].flatten().tolist() == [1, 2, 3, 4, 5], 'rows not concatenated'
    position_ids = batch['position_ids']
    assert position_ids.flatten().tolist() == [0, 1, 2, 0, 1], f'position_ids not packed: {position_ids}'
    # inherited from twinkle's InputProcessor (dev must NOT re-implement these)
    assert InputProcessor._is_packed_position_ids(position_ids) is True

    packed_params = InputProcessor._get_packed_seq_params(position_ids)
    assert packed_params.cu_seqlens_q.tolist() == [0, 3, 5], \
        f'wrong sequence boundaries: {packed_params.cu_seqlens_q}'
    assert int(packed_params.max_seqlen_q) == 3


def test_unpacked_batch_is_not_flagged_as_packed():
    """Negative control: a normal (non-packed) batch must NOT look packed.

    Guards the assertion above from being vacuously true -- if _is_packed_position_ids returned True
    for everything, the positive test would pass even with packing broken.
    """
    processor = _processor('transformers')
    prepared = processor.prepare_inputs([{'input_ids': [1, 2, 3], 'labels': [1, 2, 3]}])
    assert prepared[0]['position_ids'].flatten().tolist() == [0, 1, 2]
    assert InputProcessor._is_packed_position_ids(prepared[0]['position_ids']) is False


def test_bookkeeping_fields_dropped_and_model_inputs_passed_through():
    """prepare_inputs filters by BLACKLIST: drop swift bookkeeping, pass everything else through.

    The dev Template's _encode already builds exactly the forward kwargs a model needs (the legacy
    convention), so a whitelist would silently swallow any new/model-specific field until someone
    remembered to register it. This test pins the blacklist semantics: an unknown-but-deliberate
    key survives, while the two bookkeeping keys do not.

    `lengths` comes from template.encode(return_length=True) (packing / group_by_length) and would
    KeyError inside twinkle's padding_map-driven collate; `_labels_shifted` is the contract-14
    marker, meaningful at encode time only.
    """
    processor = _processor('transformers')
    prepared = processor.prepare_inputs([{
        'input_ids': [1, 2, 3],
        'labels': [1, 2, 3],
        'lengths': 3,
        'length': 3,
        '_labels_shifted': True,
        'token_type_ids': [0, 0, 0],  # a real model input that no whitelist would have listed
    }])
    feat = prepared[0]
    for dropped in ('lengths', 'length', '_labels_shifted'):
        assert dropped not in feat, f'{dropped} must be dropped before collate'
    assert 'token_type_ids' in feat, 'a deliberate model input must pass through the blacklist'
    assert 'input_ids' in feat and 'labels' in feat


def test_align_routed_experts_uses_padded_length_not_stale_cache():
    """routed_experts must be aligned to the POST-pad_cp sequence length.

    twinkle's pipeline runs pad_cp before align_routed_experts (processor/base.py:71-83) because
    pad_cp extends input_ids/labels to a multiple of 2*cp_size but does NOT pad routed_experts.
    With the old order routed_experts stayed short and silently misaligned with the tokens.

    The `length` below is deliberately stale: dev's prepare_inputs drops it as bookkeeping, and
    twinkle's align_routed_experts derives the target from input_ids anyway -- so neither path can
    align against a pre-pad value. Both defences are load-bearing and this asserts the outcome.
    """
    import twinkle
    from twinkle import DeviceMesh

    seq, layers, topk, cp = 5, 2, 4, 2  # 5 is not a multiple of 2*cp=4 -> pad_cp extends to 8
    # A REAL mesh, and twinkle initialized, are both required for the mesh to survive construction:
    # InputProcessor is @remote_class-decorated, and that wrapper does two different things to the
    # argument depending on global state. With no device group it strips every DeviceMesh from the
    # kwargs; with a device group but an argument that is not a DeviceMesh instance (a duck-typed
    # SimpleNamespace, as this test used to pass) it REPLACES it with twinkle's global default mesh.
    # Either way pad_cp then sees cp_world_size 1 or no mesh at all, silently becomes a no-op, and the
    # failure surfaces as an unpadded length rather than as anything about the mesh -- so assert the
    # object identity too.
    twinkle.initialize(mode='local')
    mesh = DeviceMesh.from_sizes(world_size=cp, cp_size=cp)
    processor = InputProcessor(device_mesh=mesh, framework='megatron', padding_free=True)
    assert processor.device_mesh is mesh, 'twinkle replaced or dropped the device_mesh we passed'

    names = [f.__name__ for f in processor.process_pipeline]
    assert names.index('pad_cp') < names.index('align_routed_experts') < names.index('collate_fn'), \
        f'pipeline order regressed: {names}'

    data = [{
        'input_ids': list(range(1, seq + 1)),
        'labels': list(range(1, seq + 1)),
        'length': seq,  # stale after pad_cp -- must NOT be used as the alignment target
        'routed_experts': torch.zeros(seq - 1, layers, topk, dtype=torch.long),
    }]
    for fn in processor.process_pipeline:
        data = fn(data)
        if fn.__name__ == 'align_routed_experts':
            break
    ids_len = data[0]['input_ids'].shape[-1]
    experts_len = data[0]['routed_experts'].shape[1]
    assert ids_len == 8, f'pad_cp should have padded 5 -> 8, got {ids_len}'
    assert experts_len == ids_len, f'routed_experts {experts_len} != input_ids {ids_len}'


def test_packing_derives_padding_free():
    """packing=True must DERIVE padding_free=True, not fail -- legacy sets it for the user.

    Legacy does `if self.packing: self.padding_free = True` (arguments/sft_args.py:186-189), so
    `--packing true` on its own is the normal usage and existing scripts never pass padding_free.
    The derivation is what makes packing actually pack: without padding_free the rows are
    pad-collated and packing silently does nothing.

    Scope of who this protects (measured, not assumed): scripts coming through the legacy CLI are
    ALREADY safe either way -- SftArguments.__post_init__ sets padding_free=True upstream and
    args_to_configs carries it into TemplateConfig, so a fail-fast rule would not have hit them.
    The derivation matters for the dev-native CLI and for code constructing Configs directly, where
    no god-class __post_init__ back-fills the coupled field.
    """
    from swift.dev.configs import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, validate_configs)

    dataset_config = DatasetConfig(dataset=['d'], packing=True)
    template_config = TemplateConfig(padding_free=False)
    validate_configs(
        ModelConfig(model='m'), template_config, dataset_config, TrainConfig(), DistributedConfig(), CheckpointConfig(),
        None)
    assert template_config.padding_free is True, 'packing did not derive padding_free'


def _validate(dataset_config, template_config=None):
    from swift.dev.configs import (CheckpointConfig, DistributedConfig, ModelConfig, TemplateConfig, TrainConfig,
                                   validate_configs)
    validate_configs(
        ModelConfig(model='m'), template_config or TemplateConfig(), dataset_config, TrainConfig(), DistributedConfig(),
        CheckpointConfig(), None)


class _StubTemplate:

    def __init__(self, is_multimodal: bool):
        self.model_meta = SimpleNamespace(is_multimodal=is_multimodal)


@pytest.mark.parametrize(
    'is_multimodal, kwargs, expected',
    [
        # auto (None): text -> eager, multimodal -> lazy (avoids encoding all media twice)
        (False, {}, 'eager'),
        (True, {}, 'lazy'),
        # DESIGN INVARIANT (not ordinary cases): auto must back off to eager for every feature a
        # cross-config rule inspects. That is what keeps validate_configs a pure function of the
        # Configs -- it can assume auto never produces a rule-violating value, so it only has to
        # reject EXPLICIT opt-ins. Deleting any of these `not ...` guards from _encode_mode would
        # silently break that assumption; these three rows go red first.
        (True, {
            'group_by_length': True
        }, 'eager'),
        (True, {
            'packing': True
        }, 'eager'),
        (True, {
            'cached_dataset': ['c']
        }, 'eager'),
        # explicit opt-in/out wins over the model shape
        (True, {
            'lazy_tokenize': False
        }, 'eager'),
        (False, {
            'lazy_tokenize': True
        }, 'lazy'),
        # streaming short-circuits regardless of lazy_tokenize
        (True, {
            'streaming': True
        }, 'stream'),
        (False, {
            'streaming': True,
            'lazy_tokenize': False
        }, 'stream'),
    ])
def test_encode_mode_resolution(is_multimodal, kwargs, expected):
    """_encode_mode resolves lazy_tokenize=None (auto) like legacy _init_lazy_tokenize."""
    from swift.dev.builders.dataset import _encode_mode
    from swift.dev.configs import DatasetConfig

    dataset_config = DatasetConfig(dataset=['d'], **kwargs)
    assert _encode_mode(dataset_config, _StubTemplate(is_multimodal)) == expected


def test_group_by_length_accepted_with_default_lazy_tokenize():
    """group_by_length with lazy_tokenize unset (auto) must pass: auto resolves to eager."""
    from swift.dev.configs import DatasetConfig

    _validate(DatasetConfig(dataset=['d'], group_by_length=True))


def test_group_by_length_with_explicit_lazy_tokenize_raises():
    """An EXPLICIT lazy_tokenize=True with group_by_length is a real conflict -> fail fast."""
    from swift.dev.configs import DatasetConfig

    dataset_config = DatasetConfig(dataset=['d'], group_by_length=True, lazy_tokenize=True)
    with pytest.raises(ValueError, match='lazy_tokenize'):
        _validate(dataset_config)


def test_explicit_lazy_tokenize_with_packing_raises():
    """packing needs the eager-only `lengths` column, so an explicit lazy opt-in must fail."""
    from swift.dev.configs import DatasetConfig, TemplateConfig

    dataset_config = DatasetConfig(dataset=['d'], packing=True, lazy_tokenize=True)
    with pytest.raises(ValueError, match='lazy_tokenize'):
        _validate(dataset_config, TemplateConfig(padding_free=True))


def test_explicit_lazy_tokenize_with_streaming_raises():
    """legacy rejects streaming + explicit lazy_tokenize (base_args.py:136-140)."""
    from swift.dev.configs import DatasetConfig

    dataset_config = DatasetConfig(dataset=['d'], streaming=True, lazy_tokenize=True)
    with pytest.raises(ValueError, match='lazy_tokenize'):
        _validate(dataset_config)
