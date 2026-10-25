# Copyright (c) ModelScope Contributors. All rights reserved.
"""S12 — export/merge/quantize contract tests (L1: CPU, no model forward).

The export recipes (quantize / merge_lora / convert / cached_dataset / hub / to_peft) are thin
delegates over their components; what can break silently is the ORCHESTRATION (which step runs,
with what derived output directory, in what order) and the pass-through of the user's knobs. These
tests pin exactly that, monkeypatching the component layer so each case asserts the calls the CLI
actually made. The real-artifact e2e halves (S12-01/02/06/10/...) need a GPU and live in the slow
tier separately.
"""
import json
import os

import pytest


@pytest.fixture(scope='module')
def base_model(tmp_path_factory):
    """A real local directory the parse-time checkpoint scan accepts (it resolves --model even
    when the dir carries no args.json; a nonexistent path is refused before anything else runs)."""
    d = tmp_path_factory.mktemp('base-model') / 'model'
    d.mkdir()
    return str(d)


def _configs(base_model, **overrides):
    """A minimal, valid export Config set with every step disabled by default."""
    from swift.dev.cli.export import parse_export_configs

    argv = ['--model', base_model]
    for key, value in overrides.items():
        argv += [f'--{key}', str(value)]
    return parse_export_configs(argv)


# --- S12-04: quantizer kwargs go verbatim to the component, never mirrored on QuantizeConfig --


def test_quantize_kwargs_pass_through_to_the_component(monkeypatch, tmp_path):
    """Knobs that live on QuantizeConfig ride quantize_config; twinkle-quantizer-only knobs
    (``damp_percent`` & co.) go verbatim through ``**quantizer_kwargs`` — mirroring every one on
    QuantizeConfig would drift, but the pass-through must not swallow a config field either."""
    captured = {}

    class _FakeComponent:
        def __init__(self, model_config, template_config, quantize_config, dataset_config, **kwargs):
            captured.update(model=model_config.model, quantize_config=quantize_config, kwargs=kwargs)

        def __call__(self):
            from types import SimpleNamespace
            return SimpleNamespace(path=str(tmp_path / 'q'))

    # run_quantize imports QuantizeComponent from the components package inside the function
    # body, so the patch point is the package the recipe resolves at call time.
    import swift.dev.components as components
    monkeypatch.setattr(components, 'QuantizeComponent', _FakeComponent)

    from swift.dev.config import ModelConfig, QuantizeConfig, TemplateConfig
    from swift.dev.recipe.export import run_quantize

    out = run_quantize(
        ModelConfig(model='/tmp/base-model'), TemplateConfig(),
        QuantizeConfig(quant_method='gptq', group_size=64),
        output_dir='/tmp/q-out', exist_ok=True, damp_percent=0.3)
    assert out == str(tmp_path / 'q')
    assert captured['model'] == '/tmp/base-model'
    assert captured['quantize_config'].group_size == 64  # config fields ride quantize_config
    assert captured['kwargs']['damp_percent'] == 0.3  # non-config knobs pass through verbatim
    assert captured['kwargs']['exist_ok'] is True
    assert captured['kwargs']['output_dir'] == '/tmp/q-out'
    # damp_percent is genuinely a quantizer-only knob (no QuantizeConfig mirror to drift)
    assert not hasattr(QuantizeConfig(), 'damp_percent')


# --- S12-11: merge output-directory semantics ---------------------------------------------


def test_merge_output_dir_defaults_to_adapter_merged(monkeypatch, tmp_path):
    """No output_dir -> the component derives ``{first adapter}-merged``; an existing directory is
    left untouched unless replace_if_exists (re-writing multi-GB weights by accident is expensive)."""
    adapter = tmp_path / 'ckpt'
    adapter.mkdir()
    (adapter / 'default').mkdir()

    calls = {}

    class _FakeMerge:
        def __init__(self, model_config, tuner_config, **kwargs):
            calls.update(kwargs)

        def __call__(self):
            from types import SimpleNamespace
            merged = calls.get('output_dir') or f'{adapter}-merged'
            return SimpleNamespace(path=merged)

    # MergeComponent is imported inside run_merge_lora's body from the components package.
    import swift.dev.components as components
    monkeypatch.setattr(components, 'MergeComponent', _FakeMerge)

    from swift.dev.config import ModelConfig, TemplateConfig, TunerConfig
    from swift.dev.recipe.export import run_merge_lora

    out = run_merge_lora(ModelConfig(model='/tmp/base-model'),
                         TunerConfig(adapters=[str(adapter)]), template_config=TemplateConfig())
    assert out == f'{adapter}-merged'
    assert calls['output_dir'] is None  # the default lives in the component, not the CLI
    assert calls['replace_if_exists'] is False


# --- S12-13: to_peft_format directory layout ------------------------------------------------


def _write_adapter_weights(path, tensor):
    from safetensors.torch import save_file
    import torch
    save_file(tensor, str(path), metadata={'format': 'pt'})


def test_to_peft_format_hoists_the_default_adapter(tmp_path):
    """A Swift adapter (``default/`` subdirectory) is rewritten into peft's flat layout with the
    default adapter hoisted; an already-peft directory comes back unchanged."""
    from swift.dev.recipe.export.merge_lora import run_to_peft_format
    import torch

    swift_adapter = tmp_path / 'swift-adapter'
    (swift_adapter / 'default').mkdir(parents=True)
    (swift_adapter / 'default' / 'adapter_model.safetensors')
    _write_adapter_weights(swift_adapter / 'default' / 'adapter_model.safetensors',
                           {'base_model.model.lora_A.default.weight': torch.zeros(2, 2)})
    (swift_adapter / 'default' / 'adapter_config.json').write_text('{"peft_type": "LORA"}')
    (swift_adapter / 'other').mkdir()
    _write_adapter_weights(swift_adapter / 'other' / 'adapter_model.safetensors',
                           {'base_model.model.lora_B.other.weight': torch.zeros(2, 2)})
    (swift_adapter / 'other' / 'adapter_config.json').write_text('{"peft_type": "LORA"}')

    out = run_to_peft_format(str(swift_adapter), str(tmp_path / 'peft-out'))
    assert out == str(tmp_path / 'peft-out')
    assert not os.path.exists(os.path.join(out, 'default'))
    assert os.path.isfile(os.path.join(out, 'adapter_model.safetensors'))
    assert os.path.isfile(os.path.join(out, 'adapter_config.json'))
    assert os.path.isdir(os.path.join(out, 'other'))

    # Independent oracle on the rewrite rules: the hoisted default loses both the per-adapter
    # subdirectory and the adapter-name segment of every tuner key; the source is untouched.
    from safetensors.torch import load_file
    hoisted = load_file(os.path.join(out, 'adapter_model.safetensors'))
    assert sorted(hoisted) == ['base_model.model.lora_A.weight']
    other = load_file(os.path.join(out, 'other', 'adapter_model.safetensors'))
    assert sorted(other) == ['base_model.model.lora_B.weight']
    assert os.path.isdir(swift_adapter / 'default')  # the source directory is never mutated
    with open(os.path.join(out, 'adapter_config.json'), encoding='utf-8') as stream:
        assert json.load(stream)['peft_type'] == 'LORA'

    # an already-peft directory is returned unchanged (no default/ subdir to hoist)
    peft_dir = tmp_path / 'already-peft'
    peft_dir.mkdir()
    (peft_dir / 'adapter_config.json').write_text('{"peft_type": "LORA"}')
    assert run_to_peft_format(str(peft_dir), str(tmp_path / 'noop')) == str(peft_dir)


def test_to_peft_format_rejects_non_lora_and_extra_states(tmp_path):
    """The converter refuses what the peft layout cannot represent, naming the reason."""
    from swift.dev.recipe.export.merge_lora import run_to_peft_format

    not_lora = tmp_path / 'not-lora'
    (not_lora / 'default').mkdir(parents=True)
    (not_lora / 'default' / 'adapter_config.json').write_text('{"peft_type": "PREFIX_TUNING"}')
    with pytest.raises(ValueError, match='Only LoRA'):
        run_to_peft_format(str(not_lora), str(tmp_path / 'out1'))

    extra = tmp_path / 'extra-state'
    (extra / 'default').mkdir(parents=True)
    (extra / 'default' / 'adapter_config.json').write_text('{"peft_type": "LORA"}')
    (extra / 'extra_states').mkdir()
    with pytest.raises(ValueError, match='extra state'):
        run_to_peft_format(str(extra), str(tmp_path / 'out2'))


# --- S12-14: push_to_hub passes the knobs through (network mocked by design) ---------------


def test_push_to_hub_requires_hub_model_id_and_forwards_knobs(monkeypatch, tmp_path):
    """--hub_model_id is required; folder_path / commit_message / token reach the hub layer."""
    from swift.dev.config import CheckpointConfig, DatasetConfig
    from swift.dev.recipe.export import run_push_to_hub

    src = tmp_path / 'model'
    src.mkdir()
    with pytest.raises(ValueError, match='hub_model_id'):
        run_push_to_hub(str(src), CheckpointConfig(), DatasetConfig(), 'msg')

    captured = {}

    class _FakeHub:
        def push_to_hub(self, repo_id, folder_path, token=None, private=None, revision=None,
                        commit_message=None):
            captured.update(repo_id=repo_id, folder_path=folder_path, token=token,
                            private=private, revision=revision, commit_message=commit_message)

    # hub.py imports get_hub inside the function body, so the patch point is its home module.
    monkeypatch.setattr('swift.dev.utils.hub.get_hub', lambda use_hf: _FakeHub())

    ckpt = CheckpointConfig(hub_model_id='user/model', hub_revision='v2')
    out = run_push_to_hub(str(src), ckpt, DatasetConfig(hub_token='secret'), 'release-1')
    assert out == str(src)
    assert captured['repo_id'] == 'user/model'
    assert captured['folder_path'] == str(src)
    assert captured['commit_message'] == 'release-1'
    assert captured['token'] == 'secret'
    assert captured['revision'] == 'v2'


# --- S12-20: _validate_export rejection matrix ----------------------------------------------


def test_validate_export_rejects_bits_without_method_and_method_without_bits(base_model):
    from swift.dev.cli.export import _validate_export

    with pytest.raises(ValueError, match='quantization method'):
        _validate_export(_configs(base_model, quant_bits=8))
    with pytest.raises(ValueError, match='quant_bits'):
        _validate_export(_configs(base_model, quant_method='awq'))
    # fp8 is the one method allowed to run without bits
    _validate_export(_configs(base_model, quant_method='fp8'))


def test_validate_export_rejects_calibration_free_gptq(base_model):
    from swift.dev.cli.export import _validate_export

    with pytest.raises(ValueError, match='calibration dataset'):
        _validate_export(_configs(base_model, quant_method='gptq', quant_bits=8))


def test_validate_export_rejects_direction_ambiguity_and_merge_overlap(base_model):
    from swift.dev.cli.export import _validate_export

    with pytest.raises(ValueError, match='exactly one conversion direction'):
        _validate_export(_configs(base_model, to_mcore=True, to_hf=True))
    with pytest.raises(ValueError, match='merge_lora'):
        _validate_export(_configs(base_model, merge_lora=True, to_mcore=True))


def test_validate_export_rejects_unsafe_to_hf_and_unwired_save_options(base_model):
    """S12-09: mcore→HF always writes safetensors (a pickle path is refused at run level); the
    --safe_serialization/--max_shard_size knobs are refused where they configure nothing."""
    from swift.dev.cli.export import _validate_export

    with pytest.raises(NotImplementedError, match='safetensors'):
        _validate_export(_configs(base_model, to_hf=True, safe_serialization=False))
    # save knobs on an operation they do not configure (a bare to_mcore, say)
    with pytest.raises(NotImplementedError, match='do not configure'):
        _validate_export(_configs(base_model, to_mcore=True, max_shard_size='2GB'))


# --- S12-19: the orchestration chain (which step runs, with what source) ---------------------


def test_run_export_configs_orchestrates_to_peft_merge_quant_push(monkeypatch, tmp_path, base_model):
    """The full chain: to_peft rewrites adapter[0], merge consumes it and clears the adapter list,
    quantize runs on the merged weights, push uploads the last artifact. The legacy if/elif
    contract means exactly ONE post-merge action per invocation."""
    from swift.dev.cli import export as X

    events = []

    def fake_to_peft(adapter, output_dir):
        events.append(('to_peft', adapter, output_dir))
        return str(tmp_path / 'peft')

    def fake_merge(model_config, tuner_config, **kwargs):
        events.append(('merge', model_config.model, list(tuner_config.adapters)))
        return str(tmp_path / 'merged')

    def fake_quantize(model_config, template_config, quantize_config, dataset_config, **kwargs):
        events.append(('quantize', model_config.model, kwargs.get('output_dir')))
        return str(tmp_path / 'quant')

    def fake_push(source, checkpoint_config, dataset_config, commit_message):
        events.append(('push', source))
        return 'user/model'

    monkeypatch.setattr(X, '_validate_export', lambda configs: None)
    # The recipe functions are imported inside run_export_configs's body from the recipe
    # package, so the patch points are the package attributes it resolves at call time.
    monkeypatch.setattr('swift.dev.recipe.export.run_to_peft_format', fake_to_peft)
    monkeypatch.setattr('swift.dev.recipe.export.run_merge_lora', fake_merge)
    monkeypatch.setattr('swift.dev.recipe.export.run_quantize', fake_quantize)
    monkeypatch.setattr('swift.dev.recipe.export.run_push_to_hub', fake_push)
    monkeypatch.setattr(X, '_force_eager_encoding_for_cached_dataset', lambda configs: None)
    # run_export_configs imports process_and_validate_configs from swift.dev.config inside its
    # body; the resolution happens at call time, so the patch point is the config package.
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *a, **k: None)

    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'default').mkdir()

    configs = _configs(base_model, adapters=str(adapter), to_peft_format=True, merge_lora=True,
                       quant_method='awq', quant_bits=8, push_to_hub=True, hub_model_id='user/model')
    result = X.run_export_configs(configs)

    assert [e[0] for e in events] == ['to_peft', 'merge', 'quantize', 'push']
    # to_peft rewrote adapter[0] in place; merge consumed that path and cleared the list
    assert events[1][2] == [str(tmp_path / 'peft')]
    assert configs['tuner_config'].adapters == []
    assert configs['model_config'].model == str(tmp_path / 'merged')
    # quantize ran on the MERGED weights
    assert events[2][1] == str(tmp_path / 'merged')
    # push uploaded quantize's artifact
    assert events[3][1] == str(tmp_path / 'quant')
    assert result == 'user/model'


def test_run_export_configs_post_merge_actions_are_mutually_exclusive(monkeypatch, tmp_path, base_model):
    """The legacy if/elif contract: quantize / cached_dataset / convert never run together, even
    when several flags are set (the first branch wins)."""
    from swift.dev.cli import export as X

    events = []

    monkeypatch.setattr(X, '_validate_export', lambda configs: None)
    monkeypatch.setattr(X, '_derive_output_dir', lambda configs: None)
    monkeypatch.setattr(X, '_force_eager_encoding_for_cached_dataset', lambda configs: None)
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *a, **k: None)
    monkeypatch.setattr('swift.dev.recipe.export.run_quantize',
                        lambda *a, **k: events.append('quantize') or str(tmp_path))
    monkeypatch.setattr('swift.dev.recipe.export.export_cached_dataset',
                        lambda *a, **k: events.append('cached') or str(tmp_path))
    monkeypatch.setattr('swift.dev.recipe.export.run_convert',
                        lambda *a, **k: events.append('convert') or str(tmp_path))

    configs = _configs(base_model, quant_method='awq', quant_bits=8, to_cached_dataset=True, to_hf=True)
    X.run_export_configs(configs)
    assert events == ['quantize']  # the first branch wins, silently, exactly as legacy did


# --- S12-17: lazy -> eager is forced at run level -------------------------------------------


def test_force_eager_encoding_for_cached_dataset(base_model):
    """Exporting a cached dataset must not run the tokenizer lazily (the exported rows would be
    un-encoded); the run level pins lazy_tokenize=False before the shared validate pass."""
    from swift.dev.cli.export import _force_eager_encoding_for_cached_dataset

    configs = _configs(base_model, to_cached_dataset=True)
    _force_eager_encoding_for_cached_dataset(configs)
    assert configs['dataset_config'].lazy_tokenize is False
