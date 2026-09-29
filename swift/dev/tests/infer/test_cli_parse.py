# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI parsing / dispatch tests for ``swift infer`` (no model weights, no GPU).

``parse_infer_configs`` turns argv into the atomic Configs and folds the legacy sampling spellings into
their canonical InferConfig fields; ``infer_main`` then dispatches to the interactive REPL or the
dataset pipeline. These tests pin the argv -> Config mapping (the regression surface for legacy
command parameters) and the interactive guards, using a local directory as ``--model`` so the
checkpoint-args restore never reaches the hub.
"""
import os

import pytest

from swift.dev.builders import build_engine_args
from swift.dev.cli.infer import (_derive_result_path, _guard_interactive, _interactive_dp_width,
                                 infer_main, parse_infer_configs)
from swift.dev.config import DistributedConfig, RolloutConfig


@pytest.fixture
def fake_model(tmp_path):
    """An existing local directory: ``--model`` pointing here short-circuits the hub snapshot_download
    that ``load_args_default`` would otherwise trigger for a non-existent model id."""
    d = tmp_path / 'model'
    d.mkdir()
    return str(d)


def test_parse_returns_every_config_key(fake_model):
    result = parse_infer_configs(['--model', fake_model])
    for key in ('model_config', 'template_config', 'dataset_config', 'infer_config', 'generation_config',
                'reward_config', 'multi_turn_config', 'rollout_config', 'distributed_config', 'tuner_config',
                'quantize_config', 'plugin_config', 'cli_config', 'runtime_config'):
        assert key in result, f'missing config key {key}'
    # multi_turn_config is the reward_config under a second name (the multi-turn carrier)
    assert result['multi_turn_config'] is result['reward_config']


def test_legacy_num_samples_drives_num_return_sequences(fake_model):
    """--num_samples (the CLI spelling) and num_return_sequences (the field) are one knob, two names."""
    r = parse_infer_configs(['--model', fake_model, '--num_samples', '4'])
    assert r['infer_config'].num_return_sequences == 4
    assert r['cli_config'].num_samples == 4
    # and the reverse spelling syncs back
    r2 = parse_infer_configs(['--model', fake_model, '--num_return_sequences', '6'])
    assert r2['cli_config'].num_samples == 6
    assert r2['infer_config'].num_return_sequences == 6


def test_legacy_pt_backend_alias(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--infer_backend', 'pt'])
    assert r['infer_config'].sampler == 'transformers'


def test_legacy_prm_threshold_folds_into_reward_threshold(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--prm_threshold', '0.3'])
    assert r['infer_config'].reward_threshold == 0.3
    # an explicit reward_threshold wins over the legacy spelling
    r2 = parse_infer_configs(['--model', fake_model, '--prm_threshold', '0.3', '--reward_threshold', '0.7'])
    assert r2['infer_config'].reward_threshold == 0.7


def test_legacy_sampling_batch_spellings(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--num_sampling_batch_size', '8', '--num_sampling_batches', '3'])
    assert r['infer_config'].batch_size == 8
    assert r['infer_config'].max_batches == 3


def test_padding_side_defaults_left(fake_model):
    assert parse_infer_configs(['--model', fake_model])['template_config'].padding_side == 'left'
    # an explicit value is respected
    r = parse_infer_configs(['--model', fake_model, '--padding_side', 'right'])
    assert r['template_config'].padding_side == 'right'


def test_stream_derived_from_dataset_presence(fake_model):
    # no dataset -> interactive -> stream on
    assert parse_infer_configs(['--model', fake_model])['generation_config'].stream is True
    # a dataset run -> batch sampling -> stream off
    r = parse_infer_configs(['--model', fake_model, '--dataset', '/tmp/does_not_matter.jsonl'])
    assert r['generation_config'].stream is False


def test_all_format_warns_on_ignored_best_of_n_knobs(fake_model, caplog):
    """output_format='all' stores every candidate as-is, so a best-of-n ranking knob is silently
    ignored; the CLI says so instead of dropping it."""
    import logging
    with caplog.at_level(logging.WARNING, logger='swift.dev'):
        parse_infer_configs(['--model', fake_model, '--output_format', 'all', '--n_best_to_keep', '3'])
    assert any("output_format='all' ignores" in rec.message for rec in caplog.records)


def test_derive_result_path_explicit_wins(fake_model, tmp_path):
    target = str(tmp_path / 'custom.jsonl')
    r = parse_infer_configs(['--model', fake_model, '--result_path', target])
    _derive_result_path(r)
    assert r['infer_config'].result_path == os.path.abspath(target)


def test_derive_result_path_dataset_fallback(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--dataset', '/tmp/x.jsonl'])
    _derive_result_path(r)
    path = r['infer_config'].result_path
    assert path and path.endswith('.jsonl') and os.path.isabs(path)
    assert os.path.join('result', 'model', 'infer_result') in path


def test_derive_result_path_no_dataset_no_path_stays_none(fake_model):
    r = parse_infer_configs(['--model', fake_model])
    _derive_result_path(r)
    assert r['infer_config'].result_path is None


# --- interactive guards ---------------------------------------------------------------


def test_guard_interactive_rejects_pooling_task_type():
    for task_type in ('seq_cls', 'embedding', 'reranker', 'generative_reranker'):
        with pytest.raises(ValueError, match='forward pass'):
            _guard_interactive(None, task_type)


def test_guard_interactive_rejects_multiple_dp_drivers(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError, match='single data-parallel driver'):
        _guard_interactive(None, 'causal_lm')


def test_guard_interactive_allows_single_dp(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '1')
    _guard_interactive(None, 'causal_lm')  # must not raise


def test_interactive_dp_width_local_reads_world_size(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    assert _interactive_dp_width(None) == 4
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    assert _interactive_dp_width(None) == 1


def test_interactive_dp_width_ray_reads_mesh(monkeypatch):
    """Under mode='ray' the width is the sampler mesh's data world size, not the torchrun world."""
    import swift.dev.builders as builders

    class _Mesh:
        data_world_size = 1

    monkeypatch.setattr(builders, 'build_device_mesh_if_dp', lambda dc: _Mesh())
    assert _interactive_dp_width(DistributedConfig(mode='ray')) == 1
    # ray with dp=1 (tp=N) is a single driver -> the guard allows it
    _guard_interactive(DistributedConfig(mode='ray'), 'causal_lm')


# --- infer_main dispatch --------------------------------------------------------------


def test_infer_main_dispatches_to_dataset_pipeline(fake_model, monkeypatch):
    """A dataset run calls run_infer (not the REPL), threading the parsed configs through."""
    import swift.dev.config as config_mod
    import swift.dev.recipe as recipe_mod

    captured = {}

    def fake_run_infer(*a, **k):
        captured['args'] = (a, k)
        return ['row']

    monkeypatch.setattr(config_mod, 'process_and_validate_configs', lambda *a, **k: None)
    monkeypatch.setattr(recipe_mod, 'run_infer', fake_run_infer)
    monkeypatch.setattr(recipe_mod, 'infer_cli', lambda *a, **k: pytest.fail('REPL must not run for a dataset'))

    out = infer_main(['--model', fake_model, '--dataset', '/tmp/x.jsonl', '--num_samples', '3'])
    assert out == ['row']
    _, kwargs = captured['args']
    assert kwargs['backend'] == 'transformers'
    assert kwargs['rlhf_config'] is not None  # the reward/multi-turn carrier is always threaded


def test_infer_main_dispatches_to_repl_without_dataset(fake_model, monkeypatch):
    import swift.dev.config as config_mod
    import swift.dev.recipe as recipe_mod

    called = {}
    monkeypatch.setattr(config_mod, 'process_and_validate_configs', lambda *a, **k: None)
    monkeypatch.setattr(recipe_mod, 'run_infer', lambda *a, **k: pytest.fail('pipeline must not run interactively'))
    monkeypatch.setattr(recipe_mod, 'infer_cli', lambda *a, **k: called.setdefault('ok', True))
    monkeypatch.setenv('WORLD_SIZE', '1')

    infer_main(['--model', fake_model])  # no dataset -> interactive
    assert called.get('ok') is True


# --- example-script flags land on the right Config ------------------------------------
# Each examples/v5/infer/*.sh script is replayed here flag-for-flag (minus the model id / dataset /
# CUDA env, which need the hub or a GPU). The point is the argv -> Config mapping: a flag that lands on
# the wrong Config silently does nothing at runtime, so every knob each example sets is asserted on the
# exact field its recipe reads. If these pass, the example's command line is wired correctly.


def test_tui_multigpu_flags(fake_model):
    """tui_multigpu.sh: interactive vLLM TP=2. Engine knobs -> RolloutConfig, decoding -> GenerationConfig."""
    r = parse_infer_configs([
        '--model', fake_model, '--infer_backend', 'vllm', '--vllm_tensor_parallel_size', '2',
        '--vllm_gpu_memory_utilization', '0.9', '--max_new_tokens', '2048', '--temperature', '0.7',
        '--stream', 'true',
    ])
    assert r['infer_config'].sampler == 'vllm'
    # vLLM engine parameters live on RolloutConfig, never on InferConfig/GenerationConfig
    assert r['rollout_config'].vllm_tensor_parallel_size == 2
    assert r['rollout_config'].vllm_gpu_memory_utilization == 0.9
    # decoding parameters live on GenerationConfig
    assert r['generation_config'].max_new_tokens == 2048
    assert r['generation_config'].temperature == 0.7
    assert r['generation_config'].stream is True
    # no dataset -> interactive; TP shards one engine, so the single-driver guard still passes
    assert not (r['dataset_config'].dataset or r['dataset_config'].val_dataset)
    _guard_interactive(r['distributed_config'], r['model_config'].task_type)


def test_tui_tools_multiturn_flags(fake_model):
    """tui_tools_multiturn.sh: interactive sandbox tools. tools/sandbox -> RolloutConfig, the multi-turn
    carrier (max_turns / max_trajectory_tokens) -> the reward_config, decoding -> GenerationConfig."""
    r = parse_infer_configs([
        '--model', fake_model, '--infer_backend', 'vllm', '--vllm_tensor_parallel_size', '2',
        '--vllm_gpu_memory_utilization', '0.9', '--tools', 'sandbox', '--max_turns', '8',
        '--max_trajectory_tokens', '8192', '--sandbox_num_envs', '1', '--sandbox_command_timeout', '60',
        '--max_new_tokens', '2048', '--temperature', '0.7',
    ])
    assert r['rollout_config'].tools == ['sandbox']
    assert r['rollout_config'].sandbox_num_envs == 1
    assert r['rollout_config'].sandbox_command_timeout == 60
    # an unset --sandbox_template keeps the local LocalEnv (a value would switch to an AgentEnv microVM)
    assert r['rollout_config'].sandbox_template is None
    # max_turns / max_trajectory_tokens live on the RLHFConfig, which doubles as the multi-turn carrier
    assert r['reward_config'].max_turns == 8
    assert r['reward_config'].max_trajectory_tokens == 8192
    assert r['multi_turn_config'] is r['reward_config']
    assert r['generation_config'].temperature == 0.7
    # tools + max_turns is exactly the combination infer_cli reads to switch the multi-turn rollout on
    assert bool(r['rollout_config'].tools and r['multi_turn_config'].max_turns is not None)


def test_dataset_tools_grpo_flags(fake_model):
    """dataset_tools_grpo.sh: an external tool plugin + GRPO dump over ray. external_plugins ->
    PluginConfig, tools/sandbox -> RolloutConfig, the grpo/save_rollout_tokens/num_samples knobs ->
    InferConfig, max_turns -> reward_config, mode/nproc -> DistributedConfig."""
    r = parse_infer_configs([
        '--model', fake_model, '--infer_backend', 'vllm', '--dataset', '/tmp/x.jsonl',
        '--external_plugins', 'examples/v5/infer/custom_tools.py', '--tools', 'calculator',
        '--max_turns', '8', '--max_trajectory_tokens', '8192', '--sandbox_num_envs', '4',
        '--num_samples', '8', '--output_format', 'grpo', '--save_rollout_tokens', 'true',
        '--mode', 'ray', '--nproc_per_node', '4', '--vllm_gpu_memory_utilization', '0.85',
        '--max_new_tokens', '2048', '--temperature', '1.0', '--result_path', './output/tools_rollout_grpo.jsonl',
    ])
    assert r['plugin_config'].external_plugins == ['examples/v5/infer/custom_tools.py']
    assert r['rollout_config'].tools == ['calculator']
    assert r['rollout_config'].sandbox_num_envs == 4
    assert r['reward_config'].max_turns == 8
    assert r['reward_config'].max_trajectory_tokens == 8192
    # --num_samples is the CLI spelling of num_return_sequences; the two stay in sync
    assert r['infer_config'].num_return_sequences == 8
    assert r['cli_config'].num_samples == 8
    assert r['infer_config'].output_format == 'grpo'
    assert r['infer_config'].save_rollout_tokens is True
    assert r['distributed_config'].mode == 'ray'
    assert r['distributed_config'].nproc_per_node == 4
    assert r['generation_config'].temperature == 1.0
    assert r['generation_config'].max_new_tokens == 2048
    # a grpo dump needs a real group (>= 2) and the rollout-token sidecar; both are satisfied here
    assert r['infer_config'].num_return_sequences >= 2 and r['infer_config'].save_rollout_tokens


def test_math_prm_orm_ray_flags(fake_model):
    """math_prm_orm_ray.sh: a model PRM channel + a rule ORM channel over ray. Both ``--orm`` and ``--prm``
    land on the RLHFConfig reward surface (``reward_config``) -- rules and model ids share one merged list
    per channel; num_samples/output_format -> InferConfig, mode/nproc -> DistributedConfig."""
    r = parse_infer_configs([
        '--model', fake_model, '--infer_backend', 'vllm', '--dataset', '/tmp/x.jsonl',
        '--num_samples', '8', '--output_format', 'all', '--orm', 'accuracy',
        '--prm', 'Qwen/Qwen2.5-Math-PRM-7B', '--mode', 'ray', '--nproc_per_node', '2',
        '--vllm_gpu_memory_utilization', '0.85', '--max_new_tokens', '2048', '--temperature', '1.0',
    ])
    # both reward channels are the RLHFConfig registry surface (shared with GRPO): the ORM rule and the
    # PRM model id each land on their own channel, in the order typed.
    assert r['reward_config'].orm == ['accuracy']
    assert r['reward_config'].prm == ['Qwen/Qwen2.5-Math-PRM-7B']
    assert r['infer_config'].num_return_sequences == 8
    assert r['infer_config'].output_format == 'all'
    assert r['distributed_config'].mode == 'ray'
    assert r['distributed_config'].nproc_per_node == 2


def test_temperature_owner_is_generation_config_not_rlhf(fake_model):
    """``temperature`` exists on both GenerationConfig and RLHFConfig (the GKD knob); field_owners pins it
    to GenerationConfig, so --temperature moves the decoding value and leaves the RLHF default alone."""
    r = parse_infer_configs(['--model', fake_model, '--temperature', '0.7'])
    assert r['generation_config'].temperature == 0.7
    assert r['reward_config'].temperature == 0.9  # RLHFConfig's own default, untouched


# --- build_engine_args: Config fields -> engine kwargs --------------------------------


def test_build_engine_args_vllm_strips_prefix():
    """The vllm_* RolloutConfig fields become unprefixed engine kwargs; the sglang_*/server_*/tools fields
    never leak in."""
    rollout = RolloutConfig(vllm_tensor_parallel_size=2, vllm_gpu_memory_utilization=0.85)
    args = build_engine_args('vllm', None, rollout)
    assert args['tensor_parallel_size'] == 2
    assert args['gpu_memory_utilization'] == 0.85
    assert all(not k.startswith(('vllm_', 'sglang_')) for k in args)
    # non-engine rollout fields (tools / sandbox / server) are not engine kwargs
    assert 'tools' not in args and 'sandbox_num_envs' not in args and 'server_port' not in args


def test_build_engine_args_transformers_uses_max_batch_size():
    """The transformers backend does its own batching, so the only engine knob is InferConfig.max_batch_size."""
    from swift.dev.config import InferConfig
    assert build_engine_args('transformers', InferConfig(max_batch_size=4), RolloutConfig()) == {'max_batch_size': 4}


# --- config validation guards (process_and_validate_configs surface) ------------------


def test_validate_tools_require_max_turns():
    """Naming a tool without max_turns is a single-turn rollout with no turn to call it in: rejected."""
    from swift.dev.config import RLHFConfig
    from swift.dev.config.validate import validate_rollout_config
    with pytest.raises(ValueError, match='requires a multi-turn rollout'):
        validate_rollout_config(RolloutConfig(tools=['sandbox']), RLHFConfig(max_turns=None))
    # with max_turns set the same tools pass
    validate_rollout_config(RolloutConfig(tools=['sandbox']), RLHFConfig(max_turns=4))


def test_validate_sandbox_num_envs_positive():
    from swift.dev.config.validate import validate_rollout_config
    with pytest.raises(ValueError, match='sandbox_num_envs must be >= 1'):
        validate_rollout_config(RolloutConfig(sandbox_num_envs=0), None)


def test_validate_save_rollout_tokens_rejects_client_backend():
    """The message-only 'client' teacher exposes no token IDs/logprobs, so there is nothing to persist."""
    from swift.dev.config import InferConfig
    from swift.dev.config.validate import validate_infer_config
    with pytest.raises(ValueError, match='token-capable local backend'):
        validate_infer_config(InferConfig(save_rollout_tokens=True, sampler='client'))
    # a local backend is fine
    validate_infer_config(InferConfig(save_rollout_tokens=True, sampler='transformers'))


def test_validate_arrow_store_format_rejects_resume():
    """``store_format='arrow'`` serialises the whole run once at finish and writes no checkpoint files, so
    it cannot continue a previous run: the arrow+resume pairing is rejected up front, and jsonl (the
    resumable container) still passes."""
    from swift.dev.config import InferConfig
    from swift.dev.config.validate import validate_infer_config
    with pytest.raises(ValueError, match='cannot be combined with resume'):
        validate_infer_config(InferConfig(store_format='arrow', resume=True))
    # jsonl + resume is the resumable combination, and arrow without resume is fine
    validate_infer_config(InferConfig(store_format='jsonl', resume=True))
    validate_infer_config(InferConfig(store_format='arrow', resume=False))


def test_validate_multi_turn_bounds():
    from swift.dev.config import RLHFConfig
    from swift.dev.config.validate import validate_multi_turn_config
    with pytest.raises(ValueError, match='max_turns must be >= 1'):
        validate_multi_turn_config(RLHFConfig(max_turns=0))
    with pytest.raises(ValueError, match='max_trajectory_tokens must be >= 1'):
        validate_multi_turn_config(RLHFConfig(max_turns=4, max_trajectory_tokens=0))


# --- runtime guards inside _run_generative (fire before any model/dataset load) ---------


def _generative_guard_call(infer_config, **kwargs):
    """Drive ``run_infer`` far enough to hit the _run_generative guards. They raise before load_prompt_rows
    / sampler build, so no dataset, cache, model or GPU is needed."""
    from swift.dev.config import DatasetConfig, GenerationConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.run_infer import run_infer
    return run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), backend='no', **kwargs)


def test_grpo_requires_at_least_two_candidates():
    from swift.dev.config import InferConfig
    with pytest.raises(ValueError, match='num_return_sequences >= 2'):
        _generative_guard_call(InferConfig(output_format='grpo', num_return_sequences=1, save_rollout_tokens=True))


def test_grpo_requires_save_rollout_tokens():
    """A grpo dump stores each candidate's rollout logprobs, which only exist when save_rollout_tokens
    forces logprob computation."""
    from swift.dev.config import InferConfig
    with pytest.raises(ValueError, match='save_rollout_tokens'):
        _generative_guard_call(InferConfig(output_format='grpo', num_return_sequences=4, save_rollout_tokens=False))


def test_dpo_n_best_to_keep_must_be_lt_num_return():
    """The lowest-scoring candidate becomes the rejected response, so it cannot also be a positive."""
    from swift.dev.config import InferConfig
    with pytest.raises(ValueError, match='n_best_to_keep'):
        _generative_guard_call(InferConfig(output_format='dpo', num_return_sequences=2, n_best_to_keep=2))


def test_save_rollout_tokens_needs_output_path(patch_prompt_rows):
    """The NPZ sidecar is placed beside the jsonl, so save_rollout_tokens without an output_path is rejected.
    This guard sits after the dataset load, so rows must be present to reach it."""
    from swift.dev.config import InferConfig
    patch_prompt_rows([{'messages': [{'role': 'user', 'content': 'hi'}]}])
    with pytest.raises(ValueError, match='needs an output_path'):
        _generative_guard_call(InferConfig(save_rollout_tokens=True, num_return_sequences=1, output_format='all'))
