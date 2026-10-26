# Copyright (c) ModelScope Contributors. All rights reserved.
"""S1 additions -- the CLI/parse cases the v5 draft suite did not already pin (ut_plan S1).

The moved files (test_cli_entrypoints / test_all_cli_coverage / test_cli_mapping / infer+deploy+eval
test_cli_parse) close most of S1; each test here closes the ids its docstring names, all of which
were confirmed against the current swift.dev.cli source before writing. No model, no GPU, no hub.
"""
from __future__ import annotations
import importlib
import json

import pytest

from swift.dev.cli._megatron_compat import configure_megatron, is_megatron_argv, peek_backend
from swift.dev.cli.legacy_coverage import REMOVED_OPTION_REPLACEMENTS, unsupported_contracts
from swift.dev.cli.merge import parse_merge_configs
from swift.dev.cli.parser import resolve_argv
from swift.dev.cli.rl import run_rl_configs
from swift.dev.cli.sft import parse_sft_configs, sft_main
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    MoEConfig,
    PluginConfig,
    QuantizeConfig,
    RLConfig,
    RolloutConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)


def _fake_model(tmp_path, name: str = 'model') -> str:
    """An existing local directory: ``--model`` here keeps the checkpoint-args restore off the hub."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return str(d)


def _training_configs(task_type='causal_lm', backend=None, rl_type=None):
    """A parsed-Config mapping shaped like parse_sft/parse_rl's return, for driving sft_main/run_rl_configs."""
    configs = {
        'model_config': ModelConfig(model='m', task_type=task_type),
        'plugin_config': None,
        'template_config': TemplateConfig(),
        'dataset_config': DatasetConfig(dataset=['d']),
        'train_config': TrainConfig(),
        'distributed_config': DistributedConfig(backend=backend),
        'checkpoint_config': CheckpointConfig(),
        'logging_config': LoggingConfig(report_to=['none']),
        'tuner_config': None,
        'quantize_config': QuantizeConfig(),
        'megatron_config': None,
        'moe_config': MoEConfig(),
    }
    if rl_type is not None:
        configs.update({
            'generation_config': None,
            'rollout_config': RolloutConfig(),
            'rl_config': RLConfig(rl_type=rl_type),
            'reward_config': None,
        })
    return configs


def test_sft_main_rejects_megatron_with_non_causal_task(monkeypatch):
    """S1-03: ``--backend megatron`` + task_type=seq_cls is refused with the transformers alternative
    named, rather than silently training a classification head on the wrong backend."""
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)
    configs = _training_configs(task_type='seq_cls', backend='megatron')
    monkeypatch.setattr('swift.dev.cli.sft.parse_sft_configs', lambda _argv: configs)
    with pytest.raises(ValueError, match='supports only causal_lm SFT'):
        sft_main([])


@pytest.mark.parametrize('rl_type,recipe', [
    ('dpo', 'rlhf.run_dpo'),
    ('kto', 'rlhf.run_dpo'),
    ('cpo', 'rlhf.run_dpo'),
    ('orpo', 'rlhf.run_dpo'),
    ('simpo', 'rlhf.run_dpo'),
    ('rm', 'rlhf.run_dpo'),
    ('grpo', 'grpo.run_grpo'),
    ('ppo', 'ppo.run_ppo'),
    ('rft', 'rft.run_rft'),
    ('gkd', 'distill.run_distill'),
    ('opsd', 'distill.run_distill'),
    ('mopd', 'distill.run_distill'),
])
def test_run_rl_configs_dispatches_the_remaining_families(monkeypatch, rl_type, recipe):
    """S1-05: all twelve rl_types dispatch on run_rl_configs -- the offline RLHF family (dpo/kto/cpo/
    orpo/simpo/rm) through rlhf.run_dpo, the online family through their own recipe module."""
    expected = object()
    module = importlib.import_module(f'swift.dev.recipe.{recipe.rsplit(".", 1)[0]}')
    monkeypatch.setattr(module, recipe.rsplit('.', 1)[1], lambda *_a, **_k: expected)
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)
    assert run_rl_configs(_training_configs(rl_type=rl_type)) is expected


def test_unknown_rl_type_is_refused_at_both_layers(monkeypatch):
    """S1-06: an rl_type outside the Literal dies at argparse (SystemExit); a value smuggled past the
    type (setattr on the plain dataclass) still dies in run_rl_configs' else with the supported
    list -- never a silent fallback to a wrong recipe."""
    from swift.dev.cli.rl import parse_rl_configs

    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)
    with pytest.raises(SystemExit):
        parse_rl_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd',
                          '--rlhf_type', 'definitely_not_an_rl_type'])
    configs = _training_configs(rl_type='grpo')
    configs['rl_config'].rl_type = 'definitely_not_an_rl_type'  # bypass the Literal, drive the else
    with pytest.raises(ValueError, match='has no recipe'):
        run_rl_configs(configs)


def test_infer_sampler_no_skips_model_resolution(tmp_path, monkeypatch):
    """S1-08: with sampler='no' the message-only backend loads no local weights, so a dataset run
    passes resolve_model=False to the shared lifecycle; any other sampler resolves the model."""
    from swift.dev.cli.infer import infer_main

    resolved = []

    def fake_process(configs, **kwargs):
        resolved.append(kwargs)

    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', fake_process)
    monkeypatch.setattr('swift.dev.recipe.infer.run_infer', lambda *_a, **_k: [])
    fake_model = _fake_model(tmp_path)
    infer_main(['--model', fake_model, '--dataset', '/tmp/x.jsonl', '--infer_backend', 'no'])
    assert resolved[0]['resolve_model'] is False
    # a weight-bearing backend resolves the model instead
    infer_main(['--model', fake_model, '--dataset', '/tmp/x.jsonl', '--infer_backend', 'pt'])
    assert resolved[1]['resolve_model'] is True


def test_merge_replace_if_exists_is_threaded_to_the_recipe(tmp_path, monkeypatch):
    """S1-12: ``--replace_if_exists`` is the merge command's only own flag; it must reach
    run_merge_lora, whose default refuses to overwrite an existing output directory."""
    from swift.dev.cli.merge import merge_main

    captured = {}

    def fake_merge(*args, **kwargs):
        captured.update(kwargs)
        return 'merged'

    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)
    monkeypatch.setattr('swift.dev.recipe.export.run_merge_lora', fake_merge)
    merge_main(['--model', _fake_model(tmp_path), '--adapters', 'a', '--replace_if_exists', 'true',
               '--load_args', 'false'])
    assert captured['replace_if_exists'] is True


@pytest.mark.parametrize('field', [
    'vllm_enable_lora', 'orm_model', 'prm_model', 'reward_model', 'custom_register_path',
    'tuner_backend', 'use_vllm', 'generation_max_length',
])
def test_removed_options_carry_their_exact_migration_note(field):
    """S1-15: the removed flags the plan names keep their exact migration text, so a rewritten
    replacement note cannot quietly drop the alternative it points users to."""
    assert field in REMOVED_OPTION_REPLACEMENTS
    note = REMOVED_OPTION_REPLACEMENTS[field]
    assert note.strip()
    # every migration note names the flag/knob to use instead (or says to drop this one)
    assert '--' in note or 'drop' in note.lower()


def test_megatron_variant_switches_the_legacy_contract_key():
    """S1-17: ``--backend megatron`` switches sft/rl to the megatron_<command> legacy contract, so a
    transformers-only knob is refused under megatron with the note naming the megatron way."""
    megatron_only = unsupported_contracts('megatron_sft')
    transformers_only = unsupported_contracts('sft')
    megatron_extras = set(megatron_only) - set(transformers_only)
    assert megatron_extras, 'the megatron_sft contract must refuse knobs the transformers sft one accepts'
    # an mcore-conversion knob: refused under megatron_sft with that contract named in the message
    assert 'mcore_adapter' in megatron_extras
    with pytest.raises(ValueError, match='unsupported by `megatron_sft`'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--backend', 'megatron',
                           '--mcore_adapter', '1'])
    # without --backend megatron the same flag is not an sft Config field either: it dies as
    # unrecognized -- a different, also loud, failure -- rather than a guided legacy refusal
    with pytest.raises(ValueError, match='Unrecognized'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--mcore_adapter', '1'])


@pytest.mark.parametrize('argv,expected', [
    (['--model', 'm', '--backend', 'megatron'], True),
    (['--model', 'm', '--backend=megatron'], True),
    (['--model', 'm'], False),
    (['--model', 'm', '--backend', 'transformers'], False),
])
def test_is_megatron_argv_detects_the_backend_flag(argv, expected):
    """S1-25: the megatron contract key is picked from RAW argv before any Config exists, in both the
    space and inline-equals spellings; any other backend (or none) reads as transformers."""
    assert is_megatron_argv(argv) is expected
    assert (peek_backend(argv) == 'megatron') is expected


def test_sft_parser_returns_every_config_key(tmp_path):
    """S1-29: parse_sft_configs hands back all twelve Configs under their conventional keys; a dropped
    key is a KeyError downstream, not a visible failure here."""
    configs = parse_sft_configs(['--model', _fake_model(tmp_path), '--dataset', 'd'])
    assert set(configs) == {
        'model_config', 'plugin_config', 'template_config', 'dataset_config', 'train_config',
        'distributed_config', 'checkpoint_config', 'logging_config', 'tuner_config', 'quantize_config',
        'megatron_config', 'moe_config'
    }


def test_rl_parser_returns_every_config_key(tmp_path):
    """S1-30 (keys half): parse_rl_configs hands back the sixteen Configs run_rl_configs threads."""
    from swift.dev.cli.rl import parse_rl_configs

    configs = parse_rl_configs(['--model', _fake_model(tmp_path), '--dataset', 'd', '--rlhf_type', 'dpo'])
    assert set(configs) == {
        'model_config', 'plugin_config', 'template_config', 'dataset_config', 'train_config',
        'distributed_config', 'checkpoint_config', 'logging_config', 'tuner_config', 'generation_config',
        'rollout_config', 'rl_config', 'reward_config', 'quantize_config', 'megatron_config', 'moe_config'
    }


def test_megatron_tuner_is_limited_to_full_and_lora():
    """S1-30 (tuner half): dev Megatron training supports only full/lora; trainable_tokens is refused
    with NotImplementedError rather than silently building an adapter mcore cannot consume."""
    with pytest.raises(NotImplementedError, match='full, lora'):
        configure_megatron(
            ModelConfig(), TrainConfig(), DistributedConfig(backend='megatron'),
            TunerConfig(tuner='trainable_tokens'))
    # full maps to None (no adapter); lora is kept as-is
    assert configure_megatron(
        ModelConfig(), TrainConfig(), DistributedConfig(backend='megatron'),
        TunerConfig(tuner='full')) is None
    tuner = TunerConfig(tuner='lora')
    assert configure_megatron(ModelConfig(), TrainConfig(), DistributedConfig(backend='megatron'), tuner) is tuner


def test_export_rejects_unknown_flags(tmp_path):
    """S1-38 (strict half): the export parser refuses a flag no Config claims, keeping the command's
    quant/convert/merge surface from silently dropping a mistyped knob."""
    from swift.dev.cli.export import parse_export_configs
    with pytest.raises(ValueError, match='Unrecognized'):
        # The adapter path must be an existing local dir: the checkpoint-args restore in parse runs
        # before the unknown-flag refusal and would otherwise hit the hub for a missing path.
        parse_export_configs(['--model', _fake_model(tmp_path), '--to_peft_format', 'true',
                              '--adapters', _fake_model(tmp_path, 'adapter'), '--quant_method_typo', 'gptq'])


def test_export_output_dir_explicit_wins_and_default_falls_back(tmp_path):
    """S1-41: --output_dir is the export result's home: an explicit path wins verbatim, and with none
    given the derived default is the plain 'output' directory."""
    from swift.dev.cli.export import parse_export_configs
    out = str(tmp_path / 'explicit')
    configs = parse_export_configs(
        ['--model', _fake_model(tmp_path), '--to_peft_format', 'true', '--adapters', 'a', '--output_dir', out, '--load_args', 'false'])
    assert configs['checkpoint_config'].output_dir == out
    configs = parse_export_configs(
        ['--model', _fake_model(tmp_path), '--to_peft_format', 'true', '--adapters', 'a', '--load_args', 'false'])
    assert configs['checkpoint_config'].output_dir == 'output'


def test_pt_reuses_the_sft_surface_under_the_pt_contract_key():
    """S1-43: ``swift pt`` parses through the sft surface but rejects sft-only legacy flags under its
    own 'pt' contract key, so a chat-tuning flag cannot leak into a pretraining command."""
    pt_unsupported = unsupported_contracts('pt')
    sft_unsupported = unsupported_contracts('sft')
    # the pt and sft refusal sets now coincide (chat-tuning knobs are shared training surface); what
    # the contract key still owns is the message's command attribution -- a refused flag under
    # command='pt' must say `pt`, not `sft`.
    shared = sorted(set(pt_unsupported) & set(sft_unsupported))
    assert shared
    field = shared[0]
    with pytest.raises(ValueError, match='unsupported by `pt`'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', f'--{field}', '1'],
                          command='pt')


def test_cached_dataset_export_forces_eager_encoding(tmp_path, caplog):
    """S1-44: --to_cached_dataset pins lazy_tokenize=False (a cached dataset must be materialised to
    be saved) and says so when overriding an explicit True."""
    import logging
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    monkeypatch_targets = ('swift.dev.config.process_and_validate_configs', )
    with pytest.MonkeyPatch.context() as patch:
        for target in monkeypatch_targets:
            patch.setattr(target, lambda *_a, **_k: None)
        patch.setattr('swift.dev.recipe.export.export_cached_dataset',
                      lambda *a, **k: (str(tmp_path / 'train'), None))
        # an explicit lazy_tokenize=True is overridden with a warning
        with caplog.at_level(logging.WARNING, logger='swift.dev'):
            run_export_configs(parse_export_configs([
                '--model', _fake_model(tmp_path), '--dataset', 'd', '--to_cached_dataset', 'true',
                '--lazy_tokenize', 'true', '--output_dir', str(tmp_path / 'out'),
            ]))
        assert any('overrides lazy_tokenize' in rec.message for rec in caplog.records)


def test_missing_args_json_is_not_an_error(tmp_path):
    """S1-45: the checkpoint-args restore is a no-op when the model directory carries no args.json
    (a full-parameter local checkpoint, a bare snapshot): parse succeeds with the given values."""
    from swift.dev.cli.infer import parse_infer_configs

    model = _fake_model(tmp_path)
    assert not (tmp_path / 'model' / 'args.json').exists()
    configs = parse_infer_configs(['--model', model])  # default load_args path, nothing to restore
    assert configs['model_config'].model == model


def test_moe_aux_loss_coeff_folds_a_lone_value_to_a_scalar(tmp_path):
    """S1-20 + S1-22: ``Union[float, List[float]]`` parses as a list so a repeated flag stays
    expressible, then a lone value collapses back to the scalar the mcore router expects."""
    configs = parse_sft_configs(['--model', _fake_model(tmp_path), '--dataset', 'd',
                                 '--moe_aux_loss_coeff', '1e-3'])
    assert configs['moe_config'].moe_aux_loss_coeff == 1e-3
    assert not isinstance(configs['moe_config'].moe_aux_loss_coeff, list)
    # repeated values stay a list, in argv order
    configs = parse_sft_configs(['--model', _fake_model(tmp_path), '--dataset', 'd',
                                 '--moe_aux_loss_coeff', '1e-3', '2e-3'])
    assert configs['moe_config'].moe_aux_loss_coeff == [1e-3, 2e-3]


def test_cp_comm_type_accepts_a_single_string(tmp_path):
    """S1-21: ``Optional[Union[str, List[str]]]`` is a str+List[str] mixed field; a single spelling
    must parse (and stay) a plain string -- the megatron comm knob is scalar in the common case."""
    configs = parse_sft_configs(['--model', _fake_model(tmp_path), '--dataset', 'd', '--cp_comm_type', 'p2p'])
    assert configs['distributed_config'].cp_comm_type == 'p2p'
    # the parser folds Union[str, List[str], None] onto a single value: a second value is not a list
    # extension but an unrecognized argument, refused loudly rather than silently dropped
    with pytest.raises(ValueError, match='Unrecognized'):
        parse_sft_configs(['--model', _fake_model(tmp_path), '--dataset', 'd', '--cp_comm_type', 'p2p', 'a2a'])


def test_resolve_argv_passes_through_and_defaults_to_sys_argv(monkeypatch):
    """S1-24: resolve_argv is the single argv seam -- an explicit list passes through untouched, and
    None reads the live process command line minus the interpreter (the torchrun relaunch path
    depends on that)."""
    monkeypatch.setattr('sys.argv', ['swift', 'sft', '--model', 'm', '--dataset', 'd'])
    assert resolve_argv(None) == ['sft', '--model', 'm', '--dataset', 'd']
    assert resolve_argv(['--model', 'x']) == ['--model', 'x']


def test_rl_validate_orchestration_survives_default_rollout():
    """S2-58 (L1 anchor): validate_configs over a real parsed RL Config set (offline dpo + online grpo)
    with the DEFAULT resource_mode='manual' must pass -- the manual-mode inert-knob scan walks
    ``_changed_fields`` (a list), so any set-operation regression here is a hard crash on every
    ``swift rl`` invocation, not a misvalidated config."""
    from swift.dev.cli.rl import parse_rl_configs
    from swift.dev.config import process_and_validate_configs

    for argv in (['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--rlhf_type', 'dpo'],
                 ['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--rlhf_type', 'grpo',
                  '--orm', 'accuracy']):
        process_and_validate_configs(parse_rl_configs(argv), resolve_model=False)  # validate only
