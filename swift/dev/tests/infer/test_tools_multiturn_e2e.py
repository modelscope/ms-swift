# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end multi-turn TOOL rollout through dev's real production wiring (no GPU, no weights).

The other infer tests each stop short of this seam:

* ``test_sandbox.py`` pins ``build_tool_sandbox`` / ``build_env_pool`` / ``tool_manager_for`` with the
  twinkle env classes monkeypatched -- pure wiring, nothing ever executes a tool.
* ``test_template_contract.py`` drives twinkle's ``MultiTurnRollout`` DIRECTLY with one shared
  ``ToolManager`` -- it never crosses dev's ``RolloutEngine`` / ``configure_multi_turn`` /
  ``_generate_with_envs`` per-episode path, and never runs dev's ``trajectory_to_rollout_sample`` (the
  translation layer where ``_trainable_positions`` lives).
* ``test_pipeline_fast.py`` serves candidates from a cache (``sampler='no'``), so it never samples.

This file closes that gap the way the two tool examples (``dataset_tools_grpo.sh`` / ``tui_tools_multiturn.sh``)
actually run: ``run_infer`` and ``infer_cli`` build the REAL ``RolloutEngine``, call
``build_tool_sandbox`` -> ``configure_multi_turn(env_pool, tool_plugins)``, and drive
``_generate_with_envs`` -- each episode leases a real ``LocalEnv``, gets its OWN ``ToolManager`` from the
plugin's ``build(env)``, really executes the tool, and the finished trajectory is translated by dev's
``trajectory_to_rollout_sample``. Only the sampler is scripted (canned replies through the real
template's ``concat_input_feature``), so no model forward and no GPU are needed; everything downstream of
``sample`` is production code.

That full loop is what caught a real off-by-one in ``_trainable_positions`` (output-order
``completion_mask`` intersected against input-order ``labels``): it slipped every trainable position, so
``response_token_ids`` no longer named the tokens the sampler returned and ``_sampled_token_logprobs``
rejected them -- crashing exactly the ``save_rollout_tokens`` GRPO path ``dataset_tools_grpo.sh`` runs.
``test_response_token_ids_align_with_sampled_tokens`` is the regression that pins the fix.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from twinkle.data_format.sampling import SampledSequence, SampleResponse
from twinkle.template.tools import HermesQwenParser

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
_REPO = Path(__file__).resolve().parents[4]
_CUSTOM_TOOLS = _REPO / 'examples' / 'v5' / 'infer' / 'custom_tools.py'

_PARSER = HermesQwenParser()
_OM, _CM = _PARSER.open_marker, _PARSER.close_marker
# The tool-call markup is assembled from the parser's own markers and chr(60)/chr(62) -- never typed as
# literal angle-bracket tags, which in a source string are indistinguishable from harness control markup.
_LT, _GT = chr(60), chr(62)


def _hermes_call(name, arg_key, arg_val):
    """A tool call in the markup swift's Qwen agent templates render and ``HermesQwenParser`` reads."""
    return (_OM + '\n' + _LT + f'function={name}' + _GT + '\n' + _LT + f'parameter={arg_key}' + _GT + '\n' +
            arg_val + '\n' + _LT + '/parameter' + _GT + '\n' + _LT + '/function' + _GT + '\n' + _CM)


class _ScriptedToolSampler:
    """Returns a canned tool call on the first turn and a canned answer afterwards, building each
    ``new_input_feature`` through the REAL ``DevMixin.concat_input_feature`` -- the exact object the
    ledger adopts via ``record``.

    The reply is chosen by counting assistant turns already on the pif, NOT by mutable sampler state,
    because ``_generate_with_envs`` runs the group's trajectories concurrently on one shared sampler
    (a turn counter would race). ``sample`` is declared ``_enable_continous_work`` since the multi-turn
    engine samples one trajectory per call and refuses a sampler that would slice such a batch.
    """

    def __init__(self, template, call_text, answer_text):
        self.template = template
        self.call_text = call_text
        self.answer_text = answer_text

    def sample(self, pifs, sampling_params=None, **kwargs):
        if isinstance(pifs, dict):
            pifs = [pifs]
        responses = []
        for pif in pifs:
            turns = sum(1 for m in (pif.get('messages') or []) if m.get('role') == 'assistant')
            text = self.call_text if turns == 0 else self.answer_text
            tokens = self.template.tokenizer.encode(text, add_special_tokens=False)
            logprobs = [[(tok, -0.1)] for tok in tokens]
            new_pif = self.template.concat_input_feature(pif, tokens)
            seq = SampledSequence(
                stop_reason='stop', tokens=tokens, logprobs=logprobs, decoded=text, new_input_feature=new_pif)
            responses.append(SampleResponse(sequences=[seq]))
        return responses

    sample._enable_continous_work = True

    def shutdown(self):
        pass

    def close(self):
        pass


def _fake_build_sampler(call_text, answer_text):
    """A stand-in for ``builders.build_sampler`` that scripts the sampler off the REAL template the
    recipe already built and passes in (so no weights, no GPU)."""

    def _build(model_config, *, template=None, **kwargs):
        return _ScriptedToolSampler(template, call_text, answer_text)

    return _build


@pytest.fixture(scope='module')
def template():
    """The REAL dev template over the cached tokenizer (``load_model=False``: tokenizer only, CPU-only).
    A fresh box without the tokenizer cached skips rather than fails, and every test here depends on it
    so the skip cascades."""
    from swift.dev.builders.template import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    try:
        _, proc = get_model_processor(MODEL, load_model=False)
    except Exception as exc:  # noqa: BLE001 -- an unfetchable tokenizer is not a rollout defect
        pytest.skip(f'{MODEL}: tokenizer not fetchable ({type(exc).__name__})')
    return build_template(TemplateConfig(template='qwen2_5', max_length=2048), proc)


@pytest.fixture(scope='module')
def grpo_run(template, tmp_path_factory):
    """Drive ``run_infer``'s real multi-turn tool path ONCE (the ``dataset_tools_grpo.sh`` shape) and
    share the result: one prompt, a 2-candidate GRPO group, the calculator tool loaded from
    ``--external_plugins``, and ``save_rollout_tokens`` on so the NPZ sidecar is written. ``template`` is
    requested only to inherit its skip; ``run_infer`` builds its own."""
    import swift.dev.builders as builders
    from swift.dev.config import (DatasetConfig, GenerationConfig, InferConfig, ModelConfig, PluginConfig,
                                  RLHFConfig, RolloutConfig, TemplateConfig)
    from swift.dev.recipe.run_infer import run_infer

    work = tmp_path_factory.mktemp('grpo')
    out_path = str(work / 'rollout.jsonl')
    prompt = {'messages': [{'role': 'user', 'content': 'What is 2*(3+4)? Use the calculator.'}]}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builders, 'load_prompt_rows', lambda *a, **k: [prompt])
        patch.setattr(
            builders, 'build_sampler',
            _fake_build_sampler(_hermes_call('calculator', 'expression', '2*(3+4)'), 'The answer is 14.'))
        rows = run_infer(
            ModelConfig(model=MODEL, task_type='causal_lm'),
            TemplateConfig(template='qwen2_5', max_length=2048),
            DatasetConfig(),
            InferConfig(num_return_sequences=2, batch_size=1, output_format='grpo', save_rollout_tokens=True),
            GenerationConfig(max_new_tokens=64, temperature=1.0),
            rlhf_config=RLHFConfig(max_turns=4),
            rollout_config=RolloutConfig(
                tools=['calculator'], sandbox_num_envs=1, sandbox_workspace_root=str(work / 'sandbox')),
            plugin_config=PluginConfig(external_plugins=[str(_CUSTOM_TOOLS)]),
            backend='transformers',
            distributed_config=None,
            split_dataset_ratio=0.0,
            output_path=out_path)
    return rows, out_path


def _load_npz(out_path, rel):
    with np.load(os.path.join(os.path.dirname(out_path), rel)) as archive:
        return {key: archive[key] for key in archive.files}


# --- dataset path: run_infer + calculator + grpo + sidecar (dataset_tools_grpo.sh) -------


def test_multiturn_trajectory_runs_the_real_tool_loop(grpo_run):
    """The headline: a real tool episode ran end to end. user -> assistant(tool_call) -> tool(observation)
    -> assistant(answer), where the observation is the REAL calculator plugin's result (``2*(3+4)`` ->
    ``'14'``, an int -- not ``'14.0'``), proving the per-episode ``ToolManager`` dispatched into it."""
    rows, _ = grpo_run
    assert len(rows) == 1
    row = rows[0]
    assert row['num_generations'] == 2 and len(row['all_messages']) == 2

    traj = row['all_messages'][0]
    assert [m['role'] for m in traj] == ['user', 'assistant', 'tool', 'assistant']
    call = traj[1]
    assert call['content'] == ''  # the whole turn was the call, so cleaning left nothing
    assert call['tool_calls'] == [{
        'type': 'function',
        'function': {
            'name': 'calculator',
            'arguments': {
                'expression': '2*(3+4)'
            }
        }
    }]
    # The real calculator executed, and the observation is tagged with the tool that produced it.
    assert traj[2] == {'role': 'tool', 'content': '14', 'name': 'calculator'}
    assert traj[-1] == {'role': 'assistant', 'content': 'The answer is 14.'}
    assert row['completions'] == ['The answer is 14.', 'The answer is 14.']


def test_response_token_ids_align_with_sampled_tokens(grpo_run, template):
    """Regression for the ``_trainable_positions`` off-by-one. ``response_token_ids`` must be the EXACT
    tokens the sampler returned, in input order, with one rollout logprob each. Selecting the
    output-order mask against input-order labels slipped every position, so the stored tokens no longer
    named the sampled ones -- ``_sampled_token_logprobs`` raised, and a lenient consumer would have kept
    a silently misaligned GRPO sidecar (wrong old_logps -> wrong importance ratio)."""
    rows, out_path = grpo_run
    npz = _load_npz(out_path, rows[0]['rollout_tokens'][0])
    response_token_ids = npz['response_token_ids'].tolist()
    rollout_logprobs = npz['rollout_logprobs'].tolist()

    assert response_token_ids, 'a generated trajectory must keep its trainable tokens'
    assert len(response_token_ids) == len(rollout_logprobs), 'one old_logp per response token, exactly'
    # The right tokens, not a run shifted by one: decoding them reproduces the final answer the sampler
    # actually emitted (the tool-call markup decodes to its text, the special markers drop out).
    decoded = template.tokenizer.decode(response_token_ids, skip_special_tokens=True)
    assert 'The answer is 14.' in decoded, decoded


def test_grpo_row_stores_the_group_with_null_scores(grpo_run):
    """The GRPO row contract: a stable group ``id``, the prompt in ``messages``, every candidate's text
    in ``completions`` and full trajectory in ``all_messages``, the NPZ paths in ``rollout_tokens``, and
    ``scores is None`` because no reward channel was configured -- reward is left to the training step.
    The jsonl on disk round-trips the returned row."""
    rows, out_path = grpo_run
    row = rows[0]
    assert {'id', 'messages', 'completions', 'all_messages', 'num_generations', 'scores',
            'rollout_tokens'} <= set(row)
    assert row['scores'] is None
    assert row['messages'] == [{'role': 'user', 'content': 'What is 2*(3+4)? Use the calculator.'}]
    assert len(row['rollout_tokens']) == 2

    with open(out_path, encoding='utf-8') as f:
        written = [json.loads(line) for line in f if line.strip()]
    assert len(written) == 1 and written[0]['id'] == row['id']


def test_npz_sidecar_schema(grpo_run):
    """The rollout-token sidecar carries the six-field training payload (format-agnostic, not a GRPO
    special case): full-length ``input_ids`` / ``labels`` / ``completion_mask`` and response-length
    ``response_token_ids`` / ``rollout_logprobs`` / ``response_loss_mask``. The response arrays are a
    strict subset of the full sequence (the prompt + observations are masked out)."""
    rows, out_path = grpo_run
    rel = rows[0]['rollout_tokens'][0]
    assert rel.endswith('_c0.npz') and os.path.exists(os.path.join(os.path.dirname(out_path), rel))

    npz = _load_npz(out_path, rel)
    assert set(npz) == {
        'input_ids', 'labels', 'completion_mask', 'response_token_ids', 'rollout_logprobs',
        'response_loss_mask'
    }
    full = npz['input_ids'].shape[0]
    resp = npz['response_token_ids'].shape[0]
    assert npz['labels'].shape == (full, ) and npz['completion_mask'].shape == (full, )
    assert npz['rollout_logprobs'].shape == (resp, ) and npz['response_loss_mask'].shape == (resp, )
    assert npz['input_ids'].dtype == np.int64 and npz['rollout_logprobs'].dtype == np.float32
    assert 0 < resp < full


def test_external_plugins_registers_the_calculator(grpo_run):
    """``--external_plugins`` really imported ``examples/v5/infer/custom_tools.py``: its ``@register``
    ran, so ``calculator`` resolves under the ``tool`` extension point. (That it also EXECUTED is proven
    by the ``'14'`` observation in ``test_multiturn_trajectory_runs_the_real_tool_loop``.)"""
    from swift.dev.rollout.sandbox import TOOL
    grpo_run  # the run above loaded the plugin file through PluginRegistry.load_configured
    assert 'calculator' in TOOL.entries


def test_cli_loads_external_tool_plugin_in_a_fresh_interpreter():
    """Regression for a real, example-breaking bug the in-process tests above structurally cannot see.

    ``swift infer`` loads ``--external_plugins`` inside ``process_configs`` -- BEFORE any recipe imports
    ``swift.dev.rollout.sandbox``, the module whose import declares the ``tool`` kind. A plugin doing
    ``@PluginRegistry.register('tool', 'calculator')`` therefore hit an empty ``PluginRegistry.KINDS`` and
    crashed with "plugin kind `tool` is not registered", breaking ``dataset_tools_grpo.sh`` on EVERY
    backend (the crash is in the config lifecycle, before backend selection). The in-process tests miss it
    because some earlier test in the session already imported sandbox, populating ``KINDS`` process-wide.

    A fresh interpreter reproduces the CLI's import order exactly, so this pins the fix: ``load_configured``
    must declare swift's built-in kinds itself (``_declare_builtin_kinds``) before importing a user file.
    """
    import subprocess
    import sys

    script = (
        'from swift.dev.plugin import PluginRegistry;'
        'from swift.dev.config import PluginConfig;'
        f'PluginRegistry.load_configured(PluginConfig(external_plugins=[{str(_CUSTOM_TOOLS)!r}]));'
        "print('REGISTERED' if PluginRegistry.get('tool', 'calculator') else 'MISSING')")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(_REPO / 'twinkle' / 'src'), str(_REPO)]))
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env, cwd=str(_REPO))
    assert proc.returncode == 0, proc.stderr
    assert 'REGISTERED' in proc.stdout, proc.stdout


# --- interactive path: infer_cli + sandbox + a real run_command (tui_tools_multiturn.sh) ---


def test_infer_cli_multiturn_sandbox_executes_a_real_command(template, monkeypatch, capsys, tmp_path):
    """The interactive example's defining capability: ``infer_cli`` with ``--tools sandbox`` runs the
    SAME per-episode rollout, but the tool is the env-bound ``run_command`` (``EnvTool.from_env`` over a
    real ``LocalEnv``), not a plugin that ignores its env. A real shell command writes a proof file into
    the leased workspace, so this asserts the sandbox truly executed -- not merely that the call was
    traced -- and that the REPL prints the tool trace and the final answer."""
    import builtins

    import swift.dev.builders as builders
    from swift.dev.config import GenerationConfig, ModelConfig, RLHFConfig, RolloutConfig, TemplateConfig
    from swift.dev.recipe.infer_tui import infer_cli

    proof = tmp_path / 'proof.txt'
    call = _hermes_call('run_command', 'command', f'echo executed > {proof}')
    monkeypatch.setattr(builders, 'build_sampler', _fake_build_sampler(call, 'The command printed hello.'))

    lines = ['run something', 'quit']
    state = {'i': 0}

    def fake_input(prompt=''):
        index = state['i']
        state['i'] += 1
        if index < len(lines):
            return lines[index]
        raise EOFError

    monkeypatch.setattr(builtins, 'input', fake_input)

    infer_cli(
        ModelConfig(model=MODEL, task_type='causal_lm'),
        TemplateConfig(template='qwen2_5', max_length=2048),
        GenerationConfig(max_new_tokens=64, temperature=1.0),
        backend='transformers',
        rollout_config=RolloutConfig(
            tools=['sandbox'], sandbox_num_envs=1, sandbox_workspace_root=str(tmp_path / 'sandbox')),
        multi_turn_config=RLHFConfig(max_turns=4))

    out = capsys.readouterr().out
    assert 'called tools: run_command' in out, out
    assert 'The command printed hello.' in out, out
    assert 'Bye' in out
    # The real LocalEnv ran the command in the workspace it leased -- the proof only exists if the
    # env-bound EnvTool actually dispatched, which a traced-but-unexecuted call would not produce.
    assert proof.exists() and proof.read_text().strip() == 'executed'
