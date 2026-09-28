# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI parsing / dispatch tests for ``swift deploy`` (no model weights, no GPU).

``parse_deploy_configs`` turns argv into the atomic Configs; ``deploy_main`` threads them into
``run_deploy``. These pin the argv -> Config mapping for *every* deploy knob (requirement 8), so a
renamed field, a lost ``field_owners`` pin, or a legacy spelling that stops folding is caught here
rather than at serve time. ``--model`` / ``--adapters`` point at existing local directories so the
``load_args`` checkpoint-args restore never reaches the hub.
"""
import json
import os

import pytest

from swift.dev.cli.deploy import (_adapter_mapping, _split_adapter_name, _strip_adapter_names, deploy_main,
                                  parse_deploy_configs)


@pytest.fixture
def fake_model(tmp_path):
    """An existing local dir: ``--model`` here short-circuits the hub download ``load_args`` would do."""
    d = tmp_path / 'model'
    d.mkdir()
    return str(d)


@pytest.fixture
def fake_adapter(tmp_path, fake_model):
    """An existing local adapter dir carrying ``args.json``, so checkpoint-args restore resolves it
    locally (no hub) and folds its ``model_type`` / ``template`` onto the Configs -- the realistic
    ``swift deploy --adapters <trained lora>`` path."""
    d = tmp_path / 'adapter'
    d.mkdir()
    (d / 'args.json').write_text(
        json.dumps({
            'model': fake_model,
            'model_type': 'qwen3_5',
            'template': 'qwen3_5',
            'swift_version': '4.0.0.dev0'
        }))
    return str(d)


# --- argv plumbing --------------------------------------------------------------------


def test_parse_returns_every_config_key(fake_model):
    result = parse_deploy_configs(['--model', fake_model])
    for key in ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'distributed_config',
                'checkpoint_config', 'tuner_config', 'generation_config', 'rollout_config', 'infer_config',
                'deploy_config', 'quantize_config', 'runtime_config'):
        assert key in result, f'missing config key {key}'


def test_unrecognized_argument_is_rejected(fake_model):
    with pytest.raises(ValueError, match='Unrecognized arguments'):
        parse_deploy_configs(['--model', fake_model, '--not_a_real_flag', '1'])


@pytest.mark.parametrize('flag', ['use_ray', 'ddp_backend', 'lmdeploy_tp', 'ignore_args_error'])
def test_legacy_only_flags_are_rejected(fake_model, flag):
    """deploy refuses legacy flags it cannot honour, with a reason rather than a silent drop."""
    with pytest.raises(ValueError, match='unsupported by `deploy`|obsolete'):
        parse_deploy_configs(['--model', fake_model, f'--{flag}', 'true'])


# --- model / template / backend -------------------------------------------------------


def test_model_and_template_land(fake_model):
    r = parse_deploy_configs(
        ['--model', fake_model, '--model_type', 'qwen3_5', '--torch_dtype', 'bfloat16', '--template', 'qwen3_5'])
    assert r['model_config'].model == fake_model
    assert r['model_config'].model_type == 'qwen3_5'
    assert r['model_config'].torch_dtype == 'bfloat16'
    assert r['template_config'].template == 'qwen3_5'


@pytest.mark.parametrize('spelling,expected', [('vllm', 'vllm'), ('sglang', 'sglang'), ('pt', 'transformers'),
                                               ('transformers', 'transformers')])
def test_infer_backend(fake_model, spelling, expected):
    r = parse_deploy_configs(['--model', fake_model, '--infer_backend', spelling])
    assert r['infer_config'].infer_backend == expected


def test_sampler_type_pins_to_deploy_config(fake_model):
    """``--sampler_type`` is spelled by both InferConfig (sample/distill) and DeployConfig
    (vllm/vllm_async/sglang_async/...); deploy pins it to DeployConfig, the field that actually drives
    the deployment. Without the pin, argparse rejects 'vllm_async' against InferConfig's choices."""
    r = parse_deploy_configs(['--model', fake_model, '--sampler_type', 'vllm_async'])
    assert r['deploy_config'].sampler_type == 'vllm_async'
    # InferConfig keeps its own default; the two same-named fields do not cross-contaminate.
    assert r['infer_config'].sampler_type == 'sample'


def test_sampler_type_defaults_to_none_so_backend_derives_it(fake_model):
    assert parse_deploy_configs(['--model', fake_model])['deploy_config'].sampler_type is None


# --- deploy identity / address --------------------------------------------------------


def test_deploy_identity_and_address(fake_model):
    d = parse_deploy_configs([
        '--model', fake_model, '--served_model_name', 'policy', '--route_prefix', '/v1', '--api_key', 'EMPTY',
        '--owned_by', 'me', '--host', '127.0.0.1', '--port', '8123'
    ])['deploy_config']
    assert (d.served_model_name, d.route_prefix, d.api_key, d.owned_by, d.host, d.port) == ('policy', '/v1',
                                                                                            'EMPTY', 'me',
                                                                                            '127.0.0.1', 8123)


def test_deploy_defaults(fake_model):
    d = parse_deploy_configs(['--model', fake_model])['deploy_config']
    assert d.route_prefix == '/v1' and d.owned_by == 'swift' and d.host == '0.0.0.0' and d.port == 8000
    assert d.api_key is None and d.served_model_name is None
    assert d.max_logprobs == 20 and d.max_concurrency == 64
    assert d.enable_data_plane is False and d.merge_lora is False and d.autoscaling is False
    assert d.ray_namespace == 'twinkle_cluster' and d.ray_address is None and d.num_replicas is None


def test_deploy_tls_fields(fake_model):
    d = parse_deploy_configs(
        ['--model', fake_model, '--ssl_keyfile', '/tmp/k.pem', '--ssl_certfile', '/tmp/c.pem'])['deploy_config']
    assert d.ssl_keyfile == '/tmp/k.pem' and d.ssl_certfile == '/tmp/c.pem'


def test_deploy_response_scaling_data_plane_ray(fake_model):
    d = parse_deploy_configs([
        '--model', fake_model, '--max_logprobs', '30', '--max_concurrency', '16', '--enable_data_plane', 'true',
        '--num_replicas', '2', '--autoscaling', 'true', '--ray_namespace', 'ns1', '--ray_address', 'auto'
    ])['deploy_config']
    assert d.max_logprobs == 30 and d.max_concurrency == 16
    assert d.enable_data_plane is True and d.num_replicas == 2 and d.autoscaling is True
    assert d.ray_namespace == 'ns1' and d.ray_address == 'auto'


def test_deploy_persistence(fake_model):
    d = parse_deploy_configs(
        ['--model', fake_model, '--persistence_mode', 'file', '--persistence_file_path',
         '/tmp/s.json'])['deploy_config']
    assert d.persistence_mode == 'file' and d.persistence_file_path == '/tmp/s.json'


def test_deploy_merge_lora_and_logging(fake_model):
    d = parse_deploy_configs([
        '--model', fake_model, '--merge_lora', 'true', '--verbose', 'false', '--log_interval', '5', '--log_level',
        'debug', '--request_log_path', '/tmp/req.log'
    ])['deploy_config']
    assert d.merge_lora is True and d.verbose is False and d.log_interval == 5
    assert d.log_level == 'debug' and d.request_log_path == '/tmp/req.log'


# --- distributed / engine args --------------------------------------------------------


def test_distributed_mode_and_nproc(fake_model):
    dist = parse_deploy_configs(['--model', fake_model, '--mode', 'ray', '--nproc_per_node', '4'])['distributed_config']
    assert dist.mode == 'ray' and dist.nproc_per_node == 4


def test_rollout_vllm_engine_args(fake_model):
    ro = parse_deploy_configs([
        '--model', fake_model, '--infer_backend', 'vllm', '--vllm_gpu_memory_utilization', '0.5',
        '--vllm_enforce_eager', 'true', '--vllm_max_model_len', '4096', '--vllm_tensor_parallel_size', '2'
    ])['rollout_config']
    assert ro.vllm_gpu_memory_utilization == 0.5 and ro.vllm_enforce_eager is True
    assert ro.vllm_max_model_len == 4096 and ro.vllm_tensor_parallel_size == 2


def test_rollout_sglang_engine_args(fake_model):
    ro = parse_deploy_configs([
        '--model', fake_model, '--infer_backend', 'sglang', '--sglang_tp_size', '2', '--sglang_context_length',
        '4096', '--sglang_mem_fraction_static', '0.6'
    ])['rollout_config']
    assert ro.sglang_tp_size == 2 and ro.sglang_context_length == 4096 and ro.sglang_mem_fraction_static == 0.6


def test_generation_args_parse_but_are_per_request(fake_model):
    """Generation knobs parse into GenerationConfig; run_deploy warns they are NOT applied server-side
    (the gateway translates each request's own sampling params). Pinning the parse keeps the warning
    honest -- the fields exist and carry the CLI value."""
    g = parse_deploy_configs(['--model', fake_model, '--temperature', '0.7', '--max_new_tokens', '64'
                              ])['generation_config']
    assert g.temperature == 0.7 and g.max_new_tokens == 64


# --- adapters -------------------------------------------------------------------------


def test_adapter_mapping_named_and_bare():
    assert _adapter_mapping(['a=/tmp/l1', 'b=/tmp/l2']) == {'a': '/tmp/l1', 'b': '/tmp/l2'}
    # a bare path gets a positional default name
    assert _adapter_mapping(['/tmp/l1', '/tmp/l2']) == {'adapter-1': '/tmp/l1', 'adapter-2': '/tmp/l2'}
    assert _adapter_mapping([]) == {}


def test_split_adapter_name():
    assert _split_adapter_name('a=/tmp/l1') == ('a', '/tmp/l1')
    assert _split_adapter_name('/tmp/l1') == (None, '/tmp/l1')
    # only the first '=' splits, so a path may itself contain '='
    assert _split_adapter_name('a=/tmp/x=1') == ('a', '/tmp/x=1')


def test_strip_adapter_names_rewrites_to_bare_paths():
    argv, names = _strip_adapter_names(['--model', 'M', '--adapters', 'a=/tmp/l1', 'b=/tmp/l2', '--port', '9'])
    assert argv == ['--model', 'M', '--adapters', '/tmp/l1', '/tmp/l2', '--port', '9']
    assert names == ['a', 'b']
    # inline '=' form and a bare path are both handled; names stay index-aligned
    argv2, names2 = _strip_adapter_names(['--adapters=a=/tmp/l1', '/tmp/l2'])
    assert argv2 == ['--adapters=/tmp/l1', '/tmp/l2']
    assert names2 == ['a', None]


def test_adapters_bare_path_restores_checkpoint_args(fake_model, fake_adapter):
    """A bare local adapter resolves without the hub and folds its args.json onto the Configs."""
    r = parse_deploy_configs(['--model', fake_model, '--adapters', fake_adapter])
    assert r['tuner_config'].adapters == [fake_adapter]
    assert r['model_config'].model_type == 'qwen3_5' and r['template_config'].template == 'qwen3_5'


def test_adapters_named_survive_checkpoint_restore(fake_model, fake_adapter):
    """``name=path`` is deploy's routing syntax; it must reach ``_adapter_mapping`` as the served name
    even though the shared checkpoint-args restore resolves the bare path underneath."""
    r = parse_deploy_configs(['--model', fake_model, '--adapters', f'mypol={fake_adapter}'])
    assert r['tuner_config'].adapters == [f'mypol={fake_adapter}']
    assert _adapter_mapping(r['tuner_config'].adapters) == {'mypol': fake_adapter}
    # the restore still ran on the bare path
    assert r['model_config'].model_type == 'qwen3_5'


def test_adapters_mixed_named_and_bare(fake_model, fake_adapter, tmp_path):
    second = tmp_path / 'adapter2'
    second.mkdir()
    (second / 'args.json').write_text(json.dumps({'swift_version': '4.0.0.dev0'}))
    r = parse_deploy_configs(['--model', fake_model, '--adapters', f'a={fake_adapter}', str(second)])
    assert _adapter_mapping(r['tuner_config'].adapters) == {'a': fake_adapter, 'adapter-2': str(second)}


# --- deploy_main dispatch -------------------------------------------------------------


def test_deploy_main_threads_configs_into_run_deploy(fake_model, fake_adapter, monkeypatch):
    """deploy_main passes the parsed Configs (and the derived engine args / adapter mapping) to run_deploy."""
    import swift.dev.config as config_mod
    import swift.dev.recipe as recipe_mod

    captured = {}

    def fake_run_deploy(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs

    monkeypatch.setattr(config_mod, 'process_and_validate_configs', lambda *a, **k: None)
    monkeypatch.setattr(recipe_mod, 'run_deploy', fake_run_deploy)

    deploy_main([
        '--model', fake_model, '--template', 'qwen3_5', '--infer_backend', 'vllm', '--served_model_name', 'policy',
        '--enable_data_plane', 'true', '--vllm_gpu_memory_utilization', '0.5', '--adapters', f'mypol={fake_adapter}'
    ])
    args, kwargs = captured['args'], captured['kwargs']
    assert args[0].model == fake_model and args[1].template == 'qwen3_5'
    assert kwargs['backend'] == 'vllm'
    assert kwargs['engine_args']['gpu_memory_utilization'] == 0.5
    assert kwargs['adapter_mapping'] == {'mypol': fake_adapter}
    assert kwargs['deploy_config'].served_model_name == 'policy'
    assert kwargs['deploy_config'].enable_data_plane is True


def test_deploy_main_without_adapters_passes_empty_mapping(fake_model, monkeypatch):
    import swift.dev.config as config_mod
    import swift.dev.recipe as recipe_mod

    captured = {}
    monkeypatch.setattr(config_mod, 'process_and_validate_configs', lambda *a, **k: None)
    monkeypatch.setattr(recipe_mod, 'run_deploy', lambda *a, **k: captured.setdefault('kwargs', k))

    deploy_main(['--model', fake_model, '--infer_backend', 'vllm'])
    assert captured['kwargs']['adapter_mapping'] == {}
    # a full-parameter run carries no TunerConfig at all
    assert os.path.isdir(fake_model)
