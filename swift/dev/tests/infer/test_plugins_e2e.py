# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end coverage of EVERY hand-writable plugin kind, through ``run_infer``'s real consumption points.

``examples/v5/infer/custom_plugins.py`` carries all five name-selected kinds (reward sync + reward async,
tool, model loader, dataset loader, template) in one file, and ``custom_sampler.py`` carries the sixth (a
custom sampler, selected by source). This file proves each of them actually WORKS -- not that it imports,
but that the run consumes it at the seam it is meant for -- driving the real production code on CPU:

  * reward (sync + async) + dataset + plugin loading: a real ``run_infer`` generative pass
    (``backend='no'``, cache-served candidates) whose rows come from the registered ``demo_synthetic``
    dataset loader and whose ``scores`` are the sum of the sync ``length_bonus`` and the ASYNC
    ``async_length_bonus`` rule. The async rule rides the same ``--orm`` list; ``compute_rewards_per_func``
    gathers its coroutine, so the combined score is the proof both ran.
  * plugin loading is also pinned in a FRESH interpreter (a subprocess), because the in-process registry
    is global: once any test imports the file the name stays registered, masking a load-order regression.
  * model loader + template: ``demo_qwen2`` lands in ``MODEL_MAPPING`` with its family, and ``demo_chatml``
    builds over the real tokenizer (``load_model=False``) into a template that encodes the chatml markup and
    now carries an ``agent_template`` (so it can drive tools -- the example pairs it with ``--tools``).
  * sampler: ``_resolve_sampler`` resolves the external ``custom_sampler.py`` source to its ``Sampler``
    subclass, the exact call ``build_sampler`` makes.
  * tool: a real multi-turn rollout through ``run_infer`` (the ``custom_plugins_infer.sh`` shape) with the
    example's OWN ``demo_chatml`` template and ``word_count`` tool, a scripted sampler standing in for the
    engine (no weights, no GPU). The observation is the real tool's output, so the per-episode ToolManager
    dispatched into the plugin.
  * per-channel ``parallel_spec``: ``_resolve_reward_channel`` (the function ``run_infer`` calls with
    ``orm_parallel_spec`` / ``prm_parallel_spec``) attaches the spec to a channel's reward-model slot and
    leaves a rule slot model-less, so the spec is inert there.

The tool rollout complements ``test_tools_multiturn_e2e.py`` (which runs a ``calculator`` over ``qwen2_5``):
here the point is that the EXAMPLE's own template + tool + agent_template wire together, which is what makes
``custom_plugins_infer.sh`` runnable.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from twinkle.data_format.sampling import SampledSequence, SampleResponse
from twinkle.template.tools import HermesQwenParser

from swift.dev.config import (DatasetConfig, GenerationConfig, InferConfig, ModelConfig, PluginConfig, RLHFConfig,
                              RolloutConfig, TemplateConfig)
from swift.dev.plugin import PluginRegistry
from swift.dev.recipe.run_infer import run_infer
from swift.dev.tests.infer.conftest import write_cache_file

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
_REPO = Path(__file__).resolve().parents[4]
CUSTOM_PLUGINS = _REPO / 'examples' / 'v5' / 'infer' / 'custom_plugins.py'
CUSTOM_SAMPLER = _REPO / 'examples' / 'v5' / 'infer' / 'custom_sampler.py'

# The three prompts ``demo_synthetic``'s DatasetLoader synthesises (custom_plugins.py). The loader's rows
# come back shuffled, so tests compare them as a set and key caches by prompt, never by position.
_SYNTHETIC = ('What is 2+2?', 'Name a primary colour.', 'Say hello in one word.')

_PARSER = HermesQwenParser()
_OM, _CM = _PARSER.open_marker, _PARSER.close_marker
# The tool-call markup is assembled from the parser's own markers and chr(60)/chr(62) -- never typed as
# literal angle-bracket tags, which in a source string are indistinguishable from harness control markup.
_LT, _GT = chr(60), chr(62)


def _hermes_call(name, arg_key, arg_val):
    """A tool call in the markup the 'hermes' agent_template renders and ``HermesQwenParser`` reads."""
    return (_OM + '\n' + _LT + f'function={name}' + _GT + '\n' + _LT + f'parameter={arg_key}' + _GT + '\n' +
            arg_val + '\n' + _LT + '/parameter' + _GT + '\n' + _LT + '/function' + _GT + '\n' + _CM)


@pytest.fixture(scope='module')
def example_plugins():
    """Import ``custom_plugins.py`` once through the real loader, the way ``run_infer`` does.

    Registration is a process-global side effect and idempotent, so loading here lets the registry-inspecting
    tests read ``MODEL_MAPPING`` / the template registry without each re-importing. The subprocess test below
    is deliberately NOT covered by this: it needs a clean interpreter to see the pre-load state.
    """
    PluginRegistry.load_configured(PluginConfig(external_plugins=[str(CUSTOM_PLUGINS)]))
    return CUSTOM_PLUGINS


@pytest.fixture(scope='module')
def tokenizer_processor():
    """The cached Qwen2.5-0.5B processor with ``load_model=False`` (tokenizer only: no weights, no GPU).
    A fresh box without it cached skips rather than fails, and every tokenizer-dependent test inherits that."""
    from swift.model import get_model_processor
    try:
        _, proc = get_model_processor(MODEL, load_model=False)
    except Exception as exc:  # noqa: BLE001 -- an unfetchable tokenizer is not a plugin defect
        pytest.skip(f'{MODEL}: tokenizer not fetchable ({type(exc).__name__})')
    return proc


# ---------------------------------------------------------------------
# reward (sync + async) + dataset loader + plugin loading, through run_infer
# ---------------------------------------------------------------------


def _expected_score(text: str) -> float:
    """What the two example rules must sum to for one completion: ``length_bonus`` (sync) caps len/100 at
    1.0, ``async_length_bonus`` (async) caps len/200 at 1.0, and the ORM channel adds them with unit
    weights (``normalize_rewards`` off, ``orm_channel_weight`` 1.0). Recomputing it here -- rather than
    hardcoding a number -- is the assertion that BOTH rules ran: drop the async one and every score falls to
    ``len/100`` alone."""
    return min(len(text) / 100.0, 1.0) + min(len(text) / 200.0, 1.0)


def test_reward_and_dataset_plugins_drive_the_scoring_pipeline(example_plugins, tmp_path):
    """One real ``run_infer`` generative pass wiring three plugin kinds at once:

    * ``--external_plugins`` loads the file, so ``demo_synthetic`` (dataset loader) and ``length_bonus`` /
      ``async_length_bonus`` (reward rules) all resolve by name;
    * the rows come from the registered dataset loader -- the three synthetic prompts, no hub access;
    * each candidate's ``scores`` is the sum of the SYNC and the ASYNC rule, proving ``compute_rewards_per_func``
      gathered the coroutine rather than dropping or awaiting it one-by-one.

    ``backend='no'`` serves candidates from ``cache_files``, so this stays offline while still crossing the
    real sample -> score -> emit -> write path.
    """
    cache = write_cache_file(
        str(tmp_path / 'cache.jsonl'),
        [([{
            'role': 'user',
            'content': 'What is 2+2?'
        }], ['4', 'four', 'twenty-two']),
         ([{
             'role': 'user',
             'content': 'Name a primary colour.'
         }], ['red', 'crimson', 'vermilion']),
         ([{
             'role': 'user',
             'content': 'Say hello in one word.'
         }], ['hi', 'hello', 'greetings'])])
    infer_config = InferConfig(cache_files=[cache], num_return_sequences=3, batch_size=3, output_format='all')

    results = run_infer(
        ModelConfig(task_type='causal_lm'),
        TemplateConfig(),
        DatasetConfig(dataset=['demo_synthetic']),  # the registered dataset-loader plugin, loaded for real
        infer_config,
        GenerationConfig(),
        rlhf_config=RLHFConfig(orm=['length_bonus', 'async_length_bonus']),  # one sync + one async rule
        plugin_config=PluginConfig(external_plugins=[str(CUSTOM_PLUGINS)]),
        backend='no',
        split_dataset_ratio=0.0)  # run over the whole 3-row dataset, not its (empty) eval split

    # the dataset plugin supplied exactly its three synthetic prompts
    assert len(results) == 3
    assert {row['messages'][0]['content'] for row in results} == set(_SYNTHETIC)
    for row in results:
        assert len(row['responses']) == 3
        # both rules scored every candidate, and their sum is what landed in the row
        assert row['scores'] == pytest.approx([_expected_score(text) for text in row['responses']], rel=1e-5)
        # the scores vary with length, so a real function ran rather than a constant stub
        assert len(set(round(s, 6) for s in row['scores'])) > 1


def test_reward_name_is_unresolvable_without_the_plugin_file():
    """The registry is process-global, so an in-process test cannot see the pre-load state: once any test
    imports ``custom_plugins.py`` the name stays registered. A FRESH interpreter reproduces the honest
    order and pins the contract both ways. It first loads a DIFFERENT plugin file (``custom_tools.py``)
    to declare swift's built-in kinds without defining ``length_bonus``, so the ``reward`` kind exists but
    the name is absent -- a loud miss, not a silent 0 -- and only loading ``custom_plugins.py`` resolves
    it. That is exactly what ``--external_plugins`` buys a run."""
    import subprocess
    import sys

    custom_tools = _REPO / 'examples' / 'v5' / 'infer' / 'custom_tools.py'
    script = (
        'from swift.dev.plugin import PluginRegistry;'
        'from swift.dev.config import PluginConfig;'
        f'PluginRegistry.load_configured(PluginConfig(external_plugins=[{str(custom_tools)!r}]));'
        "print('PRE:' + ('RESOLVED' if PluginRegistry.kind('reward').entries.get('length_bonus') else 'MISSING'));"
        f'PluginRegistry.load_configured(PluginConfig(external_plugins=[{str(CUSTOM_PLUGINS)!r}]));'
        "print('POST:' + ('RESOLVED' if PluginRegistry.get('reward', 'length_bonus') else 'MISSING'))")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(_REPO / 'twinkle' / 'src'), str(_REPO)]))
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env, cwd=str(_REPO))
    assert proc.returncode == 0, proc.stderr
    assert 'PRE:MISSING' in proc.stdout, proc.stdout
    assert 'POST:RESOLVED' in proc.stdout, proc.stdout


# ---------------------------------------------------------------------
# model loader + template plugins, through the real builders
# ---------------------------------------------------------------------


def test_model_loader_plugin_registers_its_family(example_plugins):
    """``@register_model`` put ``demo_qwen2`` into ``MODEL_MAPPING`` with the family it declares, so
    ``--model_type demo_qwen2`` selects THIS loader (which checkpoint ids it covers, which template it
    speaks, which class it loads) rather than falling through to an external-source lookup."""
    from swift.dev.model.loader import MODEL_MAPPING

    loader = MODEL_MAPPING.get('demo_qwen2')
    assert loader is not None and loader.__name__ == 'DemoQwen2Loader'
    assert loader.template == 'qwen2_5'
    assert loader.models == [MODEL]
    assert loader.architectures == ['Qwen2ForCausalLM']


def test_template_plugin_builds_and_is_tool_capable(example_plugins, tokenizer_processor):
    """``register_template`` wrote ``demo_chatml`` into the legacy registry ``build_template`` resolves, so
    ``--template demo_chatml`` builds a real dev template: it encodes the chatml markup, and -- because the
    TemplateMeta now declares ``agent_template='hermes'`` -- it can drive a multi-turn tool rollout (the
    example pairs it with ``--tools word_count``). Without that field ``template.agent_template`` raises, so
    this is the regression for the example's tool run."""
    from swift.dev.builders.template import build_template

    template = build_template(TemplateConfig(template='demo_chatml', max_length=2048), tokenizer_processor)
    encoded = template.encode({'messages': [{'role': 'user', 'content': 'hi'}]})
    ids = encoded['input_ids'] if isinstance(encoded, dict) else encoded
    decoded = template.tokenizer.decode(ids)
    # the exact chatml markup the TemplateMeta's prompt/chat_sep/suffix declare. The special tokens are
    # built from chr(60)/chr(62) rather than typed as literals: in this repo's terminal a literal
    # FIM-style control token in an edit is indistinguishable from harness control markup and gets mangled.
    im_start = chr(60) + '|im_start|' + chr(62)
    im_end = chr(60) + '|im_end|' + chr(62)
    assert decoded == f'{im_start}user\nhi{im_end}\n{im_start}assistant\n'

    # agent_template resolves (does not raise): the template is tool-capable.
    assert template.agent_template is not None


# ---------------------------------------------------------------------
# sampler plugin, resolved from an external source
# ---------------------------------------------------------------------


def test_sampler_plugin_resolves_from_its_source_file():
    """A custom sampler is selected by pointing ``--sampler`` AT ITS FILE, not by an ``@register`` (its kind
    is declared lazily inside ``build_sampler``). ``_resolve_sampler`` -- the exact call ``build_sampler``
    makes -- loads the source through the unified loader and picks the twinkle ``Sampler`` subclass out of
    it, with no built-in engine attached."""
    from twinkle.sampler import TransformersSampler

    from swift.dev.builders.sampler import _resolve_sampler

    cls, engine = _resolve_sampler(str(CUSTOM_SAMPLER))
    assert cls.__name__ == 'LoggingTransformersSampler'
    assert issubclass(cls, TransformersSampler)
    assert engine is None  # an external source is not one of the named built-in engines


# ---------------------------------------------------------------------
# tool plugin, through the real multi-turn rollout
# ---------------------------------------------------------------------


class _ScriptedToolSampler:
    """Returns a canned ``word_count`` tool call on the first turn and a canned answer afterwards, building
    each ``new_input_feature`` through the REAL ``concat_input_feature`` (the object the ledger adopts).

    The reply is chosen by counting assistant turns already on the pif, not by mutable sampler state,
    because the multi-turn engine samples a group's trajectories concurrently on one shared sampler.
    ``sample`` is declared ``_enable_continous_work`` since the engine samples one trajectory per call."""

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


def test_tool_plugin_runs_the_real_multiturn_loop(example_plugins, tokenizer_processor, tmp_path):
    """The example's own ``demo_chatml`` template + ``word_count`` tool drive a REAL multi-turn rollout
    through ``run_infer`` (the ``custom_plugins_infer.sh`` shape). Only the sampler is scripted, so
    everything downstream of ``sample`` is production code: the engine opens the episode, bakes the tool
    schema into the prompt via the template's ``agent_template``, parses the model's call, the per-episode
    ``ToolManager`` dispatches into the plugin's ``_WordCount``, and the observation is fed back for the
    final answer. The trajectory's tool turn carries the real word count, proving the plugin executed."""
    import swift.dev.builders as builders

    prompt = {'messages': [{'role': 'user', 'content': 'How many words are in "hello there world"?'}]}
    call = _hermes_call('word_count', 'text', 'hello there world')

    def _fake_build_sampler(model_config, *, template=None, **kwargs):
        return _ScriptedToolSampler(template, call, 'There are 3 words.')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builders, 'load_prompt_rows', lambda *a, **k: [prompt])
        patch.setattr(builders, 'build_sampler', _fake_build_sampler)
        results = run_infer(
            ModelConfig(model=MODEL, task_type='causal_lm'),
            TemplateConfig(template='demo_chatml', max_length=2048),
            DatasetConfig(),
            InferConfig(num_return_sequences=1, batch_size=1, output_format='all'),
            GenerationConfig(max_new_tokens=64, temperature=1.0),
            rlhf_config=RLHFConfig(max_turns=4),
            rollout_config=RolloutConfig(
                tools=['word_count'], sandbox_num_envs=1, sandbox_workspace_root=str(tmp_path / 'sandbox')),
            plugin_config=PluginConfig(external_plugins=[str(CUSTOM_PLUGINS)]),
            backend='transformers',
            distributed_config=None,
            split_dataset_ratio=0.0)

    assert len(results) == 1
    trajectory = results[0]['messages']
    assert [m['role'] for m in trajectory] == ['user', 'assistant', 'tool', 'assistant']
    call_turn = trajectory[1]
    assert call_turn['tool_calls'] == [{
        'type': 'function',
        'function': {
            'name': 'word_count',
            'arguments': {
                'text': 'hello there world'
            }
        }
    }]
    # the real word_count plugin ran: "hello there world" -> '3', tagged with the tool that produced it
    assert trajectory[2] == {'role': 'tool', 'content': '3', 'name': 'word_count'}
    assert trajectory[-1] == {'role': 'assistant', 'content': 'There are 3 words.'}


# ---------------------------------------------------------------------
# per-channel parallel_spec, at the seam run_infer resolves it
# ---------------------------------------------------------------------


def test_per_channel_parallel_spec_attaches_to_the_model_slot_only(example_plugins):
    """``run_infer`` threads ``orm_parallel_spec`` / ``prm_parallel_spec`` into ``_resolve_reward_channel``,
    which attaches the spec to the channel's reward-MODEL slot (so its DeviceGroup width / DeviceMesh derive
    from it) and leaves a rule slot model-less, where the spec is inert. Asserting both directions pins the
    per-channel routing: a rule-only ``--orm`` ignores the spec; a channel that resolves a model carries it."""
    from swift.dev.recipe.run_infer import _resolve_reward_channel

    rule_slots = _resolve_reward_channel(
        ['length_bonus'], None, None, 'no', parallel_spec='dp2', channel='orm')
    assert len(rule_slots) == 1 and not rule_slots[0].is_model  # a rule holds no model, so no spec to attach

    # an adapter with no model id resolves to a sampler-reusing generative judge, which DOES carry the spec
    model_slots = _resolve_reward_channel(
        [], '/tmp/reward_adapter', MODEL, 'transformers', parallel_spec='dp2', channel='prm')
    model_slot = next(slot for slot in model_slots if slot.is_model)
    assert model_slot.spec.kind == 'generative_reuse'
    assert model_slot.spec.parallel_spec == 'dp2'
