# Copyright (c) ModelScope Contributors. All rights reserved.
"""S17 — swift/dev/utils L1 tests: the naming tables, scoring kernels, env switches and hub
dispatch, each pinned against a hand-computed oracle (never a round-trip through the same table).

Already covered elsewhere and NOT repeated here: plugin registry protocol (component/plugin),
placement planning and weighted-reward metrics (feature/rl/test_scoring_utils.py), optim alias
spot checks (feature/sft/test_alignment.py -- this file pins the FULL mapping tables).
"""
import pytest


# --- 17a: the naming tables, in full ---------------------------------------------------------


def test_optim_name_map_covers_every_entry_round_trippable_and_rejects_unknown():
    """Every _OPTIM_NAME_MAP entry resolves to a torch.optim/twinkle class NAME; unknown names
    fail loudly listing what exists (a silent miss would let a typo fall back to a default)."""
    from swift.dev.utils.naming import _OPTIM_NAME_MAP, resolve_optim

    for swift_name, torch_name in _OPTIM_NAME_MAP.items():
        assert resolve_optim(swift_name) == torch_name
    with pytest.raises(NotImplementedError, match='Unknown optimizer'):
        resolve_optim('adamw_bananarama')


def test_scheduler_map_translates_each_style():
    """The megatron decay-style map: same-name styles pass through, aliased ones translate, and
    an unknown name is refused with the supported list."""
    from swift.dev.utils.naming import _MEGATRON_DECAY_STYLE_MAP, resolve_megatron_decay_style, resolve_scheduler

    for swift_name, mcore_style in _MEGATRON_DECAY_STYLE_MAP.items():
        assert resolve_megatron_decay_style(swift_name) == mcore_style, swift_name
    with pytest.raises(NotImplementedError, match='not supported on the Transformers backend'):
        resolve_scheduler('polynomial_decay')


def test_hf_attn_names_translate_and_versioned_names_fail_fast():
    """transformers-style attn names reach the Megatron surface translated; version-pinned names
    (flash_attention_2/3/4) collapse to the flash KERNEL with the pin itself enforced separately
    by twinkle's MegatronStrategy, and an unsupported pin or HF-only kernel fails fast."""
    from swift.dev.utils.naming import _HF_ONLY_ATTN_NAMES, _HF_TO_MEGATRON_ATTN, resolve_megatron_attn_backend

    for hf_name, mcore_name in _HF_TO_MEGATRON_ATTN.items():
        assert resolve_megatron_attn_backend(hf_name).name == mcore_name
    # every supported version pin lands on the flash kernel (the pin is enforced elsewhere)
    for pinned in ('flash_attention_2', 'flash_attention_3', 'flash_attention_4'):
        assert resolve_megatron_attn_backend(pinned).name == 'flash'
    # an unsupported pin, the HF-only kernels and nonsense all fail fast
    for refused in ('flash_attention_5', *_HF_ONLY_ATTN_NAMES, 'nonsense'):
        with pytest.raises(NotImplementedError):
            resolve_megatron_attn_backend(refused)


def test_strategy_map_is_total_over_its_names():
    from swift.dev.utils.naming import _STRATEGY_NAME_MAP, resolve_strategy

    for swift_name, twinkle_name in _STRATEGY_NAME_MAP.items():
        assert resolve_strategy(swift_name) == twinkle_name
    with pytest.raises(NotImplementedError):
        resolve_strategy('colossal')


def test_parse_optim_args_hand_computed():
    """k=v pairs split on commas with per-value type coercion -- hand-computed, including the
    falsy spellings and nested quotes that a naive json.loads would trip on."""
    from swift.dev.utils.naming import parse_optim_args

    assert parse_optim_args(None) == {}
    assert parse_optim_args('lr=1e-3,beta=0.9') == {'lr': 1e-3, 'beta': 0.9}
    assert parse_optim_args('weight_decay=0.01,amsgrad=true,no_grad=false,eps=None,name=x') == {
        'weight_decay': 0.01, 'amsgrad': True, 'no_grad': False, 'eps': None, 'name': 'x'}


def test_resolve_loss_maps_and_rejects():
    from swift.dev.utils.naming import resolve_loss

    assert resolve_loss('cross_entropy') is not None
    with pytest.raises(NotImplementedError):
        resolve_loss('bananarama_loss')


# --- 17b: scoring kernels with hand-computed oracles ------------------------------------------


def _feature(labels, completion_mask=None):
    """One scored row in the shifted layout: labels carry -100 over the prompt."""
    feat = {'labels': list(labels)}
    if completion_mask is not None:
        feat['completion_mask'] = list(completion_mask)
    return feat


def test_response_positions_picks_the_scored_span():
    """Positions are where labels != -100 -- the completion span, hand-computed for a prompt of
    3 masked tokens followed by a 4-token response."""
    from swift.dev.utils.scoring import response_positions

    feat = _feature([-100, -100, -100, 5, 6, 7, 8])
    assert response_positions(feat) == [3, 4, 5, 6]
    # an all-masked row (prompt-only) has no response at all
    assert response_positions(_feature([-100, -100])) == []


def test_row_token_scores_gathers_per_position_values():
    """One score per response position, taken from the row's per-token column in order."""
    import torch

    from swift.dev.utils.scoring import row_token_scores

    feat = _feature([-100, -100, -100, 5, 6, 7, 8])
    row = torch.tensor([1., 2., 3., 4., 5., 6., 7.])  # exact in float32
    assert row_token_scores(row, [3, 4, 5, 6], feat) == [4.0, 5.0, 6.0, 7.0]


def test_resolve_galore_target_variants_and_refusals():
    """GaLore upgrades ONLY the bases that have a variant: AdamW->GaLoreAdamW, the HF Adafactor
    CLASS (what resolve_optim_target returns for 'adafactor') -> GaLoreAdafactor, while Adam /
    SGD / Muon are refused rather than silently training an unprojected optimizer. The quantized
    QGaLore is AdamW-only; quantize+Adafactor is refused instead of dropping the quantization."""
    from transformers.optimization import Adafactor

    from swift.dev.utils.naming import resolve_galore_target

    assert resolve_galore_target('AdamW') == 'GaLoreAdamW'
    assert resolve_galore_target(Adafactor) == 'GaLoreAdafactor'  # the class form, keyed off __name__
    assert resolve_galore_target('AdamW', quantize=True) == 'QGaLoreAdamW8bit'
    with pytest.raises(NotImplementedError, match='no GaLore variant'):
        resolve_galore_target('SGD')
    with pytest.raises(NotImplementedError, match='no QGaLore variant'):
        resolve_galore_target(Adafactor, quantize=True)
    with pytest.raises(NotImplementedError, match='no QGaLore variant'):
        resolve_galore_target('Adam', quantize=True)


def test_response_token_scores_aligns_batched_frame():
    """The whole-batch frame -> one score list per feature, each row indexed at its OWN response
    positions so right-padding never shifts a feature's scores; a missing output raises unless
    allow_missing (PPO's no-KL read), and a row-count mismatch is refused loudly."""
    import torch

    from swift.dev.utils.scoring import response_token_scores

    # two features: response at [2,3] and [1,2,3]; the frame is right-padded to length 5.
    # values are exact in float32 (the kernel casts through .float())
    feats = [{'labels': [-100, -100, 5, 6]}, {'labels': [-100, 4, 5, 6]}]
    frame = {'logps': torch.tensor([[9., 9., 1., 2., 9.], [9., 3., 4., 5., 9.]])}
    out = response_token_scores(frame, feats)
    assert out == [[1.0, 2.0], [3.0, 4.0, 5.0]]
    # a dict frame with a different output_key (a value / seq_cls head reads 'logits')
    logits_frame = {'logits': torch.zeros(2, 5)}
    assert response_token_scores(logits_frame, feats, output_key='logits') == [[0.0, 0.0], [0.0, 0.0, 0.0]]
    # response-only rows (numel == len(positions)) pass through without indexing; this form
    # only stacks when the responses are equal-length, so use two 2-token responses
    equal = [{'labels': [-100, -100, 5, 6]}, {'labels': [-100, -100, 4, 5]}]
    short = {'logps': torch.tensor([[1., 2.], [3., 4.]])}
    assert response_token_scores(short, equal) == [[1.0, 2.0], [3.0, 4.0]]
    # a 1-D frame is one row for one feature
    assert response_token_scores(torch.tensor([0., 0., 1., 2.]), [feats[0]]) == [[1., 2.]]
    # missing key: refused by default, [None]*N under allow_missing
    with pytest.raises(RuntimeError, match="no 'logps'"):
        response_token_scores({}, feats)
    assert response_token_scores({}, feats, allow_missing=True) == [None, None]
    # wrong row count: alignment is impossible and must not be guessed
    with pytest.raises(RuntimeError, match='rows for'):
        response_token_scores(torch.zeros(3, 5), feats)


def test_copy_inputs_roundtrips_and_copies():
    """copy_inputs clones the batch's tensor-bearing fields so an overlapped scorer cannot mutate
    the trainer's copy; the clone compares equal but is NOT the same object."""
    import torch

    from swift.dev.utils.scoring import copy_inputs

    batch = [{'input_ids': torch.tensor([1, 2]), 'labels': torch.tensor([-100, 2])}]
    cloned = copy_inputs(batch)
    assert len(cloned) == 1
    assert torch.equal(cloned[0]['input_ids'], batch[0]['input_ids'])
    assert cloned[0]['input_ids'] is not batch[0]['input_ids']
    # mutating the clone leaves the original untouched -- the point of the copy
    cloned[0]['input_ids'][0] = 999
    assert batch[0]['input_ids'][0].item() == 1


# --- 17c: env switches and hub dispatch -------------------------------------------------------


def test_use_hf_hub_reads_the_env(monkeypatch):
    from swift.dev.utils import env as env_utils

    for raw, expected in (('1', True), ('0', False), ('true', True), ('FALSE', False)):
        monkeypatch.setenv('USE_HF', raw)
        # _strtobool yields 0/1 (an int), truthy iff the flag is on
        assert bool(env_utils.use_hf_hub()) is expected, raw


def test_hf_endpoint_override(monkeypatch):
    from swift.dev.utils import env as env_utils

    monkeypatch.setenv('HF_ENDPOINT', 'https://hf-mirror.com')
    assert env_utils.get_hf_endpoint() == 'https://hf-mirror.com'


def test_dist_setting_and_master_from_env(monkeypatch):
    """The classic torchrun quartet (RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE) drives
    get_dist_setting, and is_master is rank==0."""
    from swift.dev.utils import env as env_utils

    monkeypatch.setenv('RANK', '3')
    monkeypatch.setenv('WORLD_SIZE', '8')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '4')
    assert env_utils.get_dist_setting() == (3, 1, 8, 4)  # rank, local_rank, world, local_world
    assert env_utils.is_master() is False
    assert env_utils.is_dist() is True
    monkeypatch.setenv('RANK', '0')
    assert env_utils.is_master() is True


def test_strtobool_table_and_env_readers(monkeypatch):
    """The boolean env primitive accepts the full y/yes/t/true/on/1 (and negative) set and raises
    on anything else; the readers on top of it (RANDOM_DATASET_RESPONSE, LOG_ONLY_MASTER) and the
    level/flags with defaults (LOG_LEVEL upper-cased, LOG_ONLY_MASTER defaulting true)."""
    from swift.dev.utils import env as env_utils

    assert env_utils._strtobool('YES') == 1 and env_utils._strtobool('off') == 0
    with pytest.raises(ValueError, match='invalid truth value'):
        env_utils._strtobool('maybe')

    monkeypatch.setenv('RANDOM_DATASET_RESPONSE', '1')
    assert env_utils.random_dataset_response() is True
    monkeypatch.delenv('RANDOM_DATASET_RESPONSE')
    assert env_utils.random_dataset_response() is False

    monkeypatch.delenv('LOG_LEVEL', raising=False)
    assert env_utils.log_level() == 'INFO'
    monkeypatch.setenv('LOG_LEVEL', 'debug')
    assert env_utils.log_level() == 'DEBUG'
    monkeypatch.delenv('LOG_ONLY_MASTER', raising=False)
    assert env_utils.log_only_master() is True
    monkeypatch.setenv('LOG_ONLY_MASTER', 'false')
    assert env_utils.log_only_master() is False


def test_dataset_offline_precedence_and_lenient_parsing(monkeypatch):
    """HF_DATASETS_OFFLINE wins over HF_HUB_OFFLINE; once set to a parseable value the fallback is
    not consulted; unset/empty -> False; a malformed value is False (never aborts the run)."""
    from swift.dev.utils import env as env_utils

    monkeypatch.delenv('HF_DATASETS_OFFLINE', raising=False)
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    assert env_utils.dataset_offline() is False
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    assert env_utils.dataset_offline() is True  # the fallback is consulted when unset
    monkeypatch.setenv('HF_DATASETS_OFFLINE', '0')
    assert env_utils.dataset_offline() is False  # precedence: datasets flag decides, hub flag ignored
    monkeypatch.delenv('HF_HUB_OFFLINE')
    monkeypatch.setenv('HF_DATASETS_OFFLINE', '')
    assert env_utils.dataset_offline() is False  # empty means unset (both flags now unset/empty)
    monkeypatch.setenv('HF_DATASETS_OFFLINE', 'typo')
    assert env_utils.dataset_offline() is False  # lenient: a typo never aborts a run


def test_get_hub_dispatches_on_use_hf():
    """get_hub returns the Hub CLASS (all methods are classmethods) picked by the flag: the
    explicit argument decides, None follows the USE_HF env."""
    from swift.dev.utils.hub import HFHub, MSHub, get_hub

    assert get_hub(False) is MSHub
    assert get_hub(True) is HFHub
    # use_hf=None follows the env flag (patched where hub.py bound it, not in env.py)
    import swift.dev.utils.hub as hub_mod

    original = hub_mod.use_hf_hub
    try:
        hub_mod.use_hf_hub = lambda: False
        assert get_hub() is MSHub
        hub_mod.use_hf_hub = lambda: True
        assert get_hub() is HFHub
    finally:
        hub_mod.use_hf_hub = original


def test_safe_snapshot_download_local_path_short_circuits(monkeypatch, tmp_path):
    """A directory that exists on disk IS the model dir -- no hub call is ever made (the download
    method is monkeypatched to raise, proving the short-circuit); a nonexistent ABSOLUTE path is
    refused rather than silently fabricated."""
    from swift.dev.utils import hub as hub_mod

    def _raising_download(*args, **kwargs):
        raise AssertionError('a local path must never reach the hub download layer')

    monkeypatch.setattr(hub_mod.MSHub, 'download_model', _raising_download)
    cached = tmp_path / 'cached-model'
    cached.mkdir()
    resolved = hub_mod.safe_snapshot_download(str(cached), use_hf=False)
    assert resolved == str(cached)
    with pytest.raises(ValueError, match='not found'):
        hub_mod.safe_snapshot_download('/nonexistent/absolute/path', use_hf=False)


def test_logger_is_namespaced_and_shared():
    """dev has exactly ONE logger, named 'swift.dev' (distinct from legacy's 'swift' so records
    are never emitted twice); every get_logger() call returns that same object, and the
    once-helpers (info_once/warning_once) are attached to it."""
    from swift.dev.utils.logger import get_logger

    a = get_logger()
    b = get_logger()
    assert a is b, 'get_logger must return the shared swift.dev logger'
    assert a.name == 'swift.dev'
    assert a.propagate is False  # keeps dev's handler off the shared legacy 'swift' logger
    assert callable(a.info_once) and callable(a.warning_once)


def test_logger_level_follows_env_and_master_only_quiets_non_master(monkeypatch):
    """LOG_LEVEL names the level dev's loggers emit at; LOG_ONLY_MASTER=true (the default) keeps
    non-master ranks at ERROR. Both read through env accessors, and the effective level mapping is
    the stdlib name -> numeric one."""
    import logging

    from swift.dev.utils import env as env_utils

    monkeypatch.setenv('LOG_LEVEL', 'debug')
    assert env_utils.log_level() == 'DEBUG'
    monkeypatch.setenv('LOG_LEVEL', 'WARNING')
    assert env_utils.log_level() == 'WARNING'
    # every name LOG_LEVEL can carry must resolve on stdlib logging (that is how the logger
    # turns it into a numeric level: getattr(logging, name, INFO))
    for name in ('DEBUG', 'INFO', 'WARNING', 'ERROR'):
        assert getattr(logging, name) > 0, name
    # the master-only contract: default true; false admits every rank at LOG_LEVEL
    monkeypatch.delenv('LOG_ONLY_MASTER', raising=False)
    assert env_utils.log_only_master() is True


# --- 17d: utils.py misc pieces with hand-computed oracles --------------------------------------


def test_add_version_to_work_dir_nests_and_counts(tmp_path):
    """Repeated runs get v0-, v1-, v2- sub-folders (never overwrite); non-vX names are ignored by
    the scan. The versioning helper only NAMES the sub-folder, so each run must materialize it
    (makedirs, as the real training driver does) for the next call to see it."""
    import os

    from swift.dev.utils.utils import add_version_to_work_dir

    first = add_version_to_work_dir(str(tmp_path))
    assert 'v0-' in first
    os.makedirs(first)
    second = add_version_to_work_dir(str(tmp_path))
    assert 'v1-' in second and second != first
    os.makedirs(second)
    (tmp_path / 'not-a-version.txt').write_text('ignored by the v(\\d+) scan')
    third = add_version_to_work_dir(str(tmp_path))
    assert 'v2-' in third


def test_append_to_jsonl_appends_rows_and_creates_parents(tmp_path):
    """One row or a list of rows, appended after existing content (the 'a' contract), parent
    directories created on demand."""
    import json as _json

    from swift.dev.utils.utils import append_to_jsonl

    fpath = str(tmp_path / 'sub' / 'rows.jsonl')
    append_to_jsonl(fpath, {'a': 1})
    append_to_jsonl(fpath, [{'b': 2}, {'b': 3}])
    with open(fpath, encoding='utf-8') as f:
        rows = [_json.loads(line) for line in f]
    assert rows == [{'a': 1}, {'b': 2}, {'b': 3}]


def test_check_json_format_masks_tokens_and_reprs_oddballs():
    """The jsonl-safety fold: scalars pass through, bool stays bool, bytes are masked, torch
    objects are stripped to a bare name, and any key containing '_token' (token_safe) is nulled so
    a serialized row never leaks credentials."""
    import torch

    from swift.dev.utils.utils import check_json_format

    assert check_json_format(3) == 3 and check_json_format('x') == 'x'
    assert check_json_format(True) is True  # bool is a subclass of int: identity, not str
    assert check_json_format(b'raw') == '<<<bytes>>>'
    assert check_json_format(torch.float16) == 'float16'
    assert check_json_format(torch.device('cpu')) == 'cpu'
    row = {'access_token': 'secret', 'nested': {'refresh_token': 's2', 'ok': 1}}
    out = check_json_format(row)
    assert out['access_token'] is None and out['nested']['refresh_token'] is None and out['nested']['ok'] == 1
    # token_safe=False keeps the keys
    kept = check_json_format({'access_token': 'secret'}, token_safe=False)
    assert kept['access_token'] == 'secret'
    # anything else degrades to a repr string (a function, say), never raising
    assert 'function' in check_json_format(check_json_format)


def test_split_str_parts_by_literal_and_regex_modes():
    """The response-splitting kernel in both modes, hand-computed: literal prefixes pair with the
    body that follows; regex mode matches the delimiter pattern against each part."""
    from swift.dev.utils.utils import split_str_parts_by
    parts = split_str_parts_by('<tag>abc</tag>', ['<tag>', '</tag>'])
    assert parts == [
        {'key': '<tag>', 'content': 'abc'},
        {'key': '</tag>', 'content': ''},
    ]
    # no delimiters -> the whole text is one unlabeled part
    assert split_str_parts_by('plain', []) == [{'key': '', 'content': 'plain'}]
    # regex mode: the pattern matches at the START of the following part
    regex_parts = split_str_parts_by('a1b22c', [r'\d+'], regex_mode=True)
    assert all(set(p) == {'key', 'content'} for p in regex_parts)
    assert ''.join(p['content'] for p in regex_parts) == 'a1b22c'


def test_split_list_contiguous_and_strided():
    """The sharding helper: contiguous splits at even linspace boundaries (3 items -> 1+1+1, 4 -> 2+2);
    strided deals round-robin. Empty shards keep the count."""
    from swift.dev.utils.utils import split_list

    assert split_list([1, 2, 3], 3) == [[1], [2], [3]]
    assert split_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    assert split_list([1, 2, 3, 4, 5], 2, contiguous=False) == [[1, 3, 5], [2, 4]]


def test_deepspeed_spec_resolution_and_zero_stage(tmp_path):
    """A preset name resolves to the bundled JSON (and its stage reads back), an inline dict passes
    through, an inline JSON string parses, a path reads the file, and None/'' -> None (off)."""
    import json as _json

    from swift.dev.utils.utils import deepspeed_zero_stage, resolve_deepspeed_config

    assert resolve_deepspeed_config(None) is None and resolve_deepspeed_config('') is None
    inline = {'zero_optimization': {'stage': 2}}
    assert resolve_deepspeed_config(inline) is inline
    assert deepspeed_zero_stage(inline) == 2
    assert deepspeed_zero_stage('zero3') == 3  # the bundled preset, read back from disk
    assert deepspeed_zero_stage({'train_batch_size': 8}) is None  # no zero_optimization -> unset
    path = tmp_path / 'ds.json'
    path.write_text(_json.dumps({'zero_optimization': {'stage': 1}}))
    assert deepspeed_zero_stage(str(path)) == 1


def test_validate_tags_uses_the_controlled_vocabulary():
    """A controlled tag passes; an unknown spelling is refused naming the offender and the
    registration context -- the whole point is stopping spelling drift at the registration site."""
    from swift.dev.utils.tags import CONTROLLED_TAGS, validate_tags

    validate_tags(['sft', 'multilingual'], context='fam-x')
    assert 'sft' in CONTROLLED_TAGS and 'CoT' not in CONTROLLED_TAGS  # canonical is lower-case
    with pytest.raises(ValueError, match="fam-x: unknown tag"):
        validate_tags(['sft', 'CoT'], context='fam-x')


# --- 17e: naming vs legacy -- the anti-drift reconciliation anchor -----------------------------


def test_naming_tables_agree_with_the_ecosystem_names():
    """S17-20: the swift-side names dev accepts must stay inside the names the wider ecosystem
    spells the same way -- every _OPTIM_NAME_MAP key is a real transformers OptimizerNames value
    (plus dev's deliberate 'adamw'/'adam' spell-throughs and the twinkle 'muon'), and every
    _SCHED_NAME_MAP key is a real transformers SchedulerType value. A renamed-or-dropped upstream
    alias turns this red and forces a deliberate decision instead of a silent dead key."""
    from transformers.training_args import OptimizerNames, SchedulerType

    from swift.dev.utils.naming import _OPTIM_NAME_MAP, _SCHED_NAME_MAP

    hf_optims = {o.value for o in OptimizerNames}
    hf_scheds = {s.value for s in SchedulerType}
    for name in _OPTIM_NAME_MAP:
        assert name in hf_optims or name in {'adamw', 'adam', 'muon'}, \
            f'{name!r} is neither an upstream optim name nor a declared dev extension'
    for name in _SCHED_NAME_MAP:
        assert name in hf_scheds, f'{name!r} is not an upstream scheduler name'
    # the load-bearing mainstream names are all present (dropping one is a silent CLI regression)
    for must in ('adamw_torch', 'adamw_torch_fused', 'adafactor', 'sgd'):
        assert must in _OPTIM_NAME_MAP
    for must in ('cosine', 'linear', 'constant', 'cosine_with_min_lr', 'inverse_sqrt'):
        assert must in _SCHED_NAME_MAP
    # the megatron decay-style values stay inside Megatron's legal set
    from swift.dev.utils.naming import _MEGATRON_DECAY_STYLE_MAP
    assert set(_MEGATRON_DECAY_STYLE_MAP.values()) <= {'cosine', 'linear', 'constant', 'inverse-square-root'}
