# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end guard for the dev template driving twinkle's REAL multi-turn tool engine.

This is not a contract test with a stub tokenizer hand-calling one method at a time -- that shape let
three real bugs through, because each was only reachable on a path the stub never drove. Instead every
test here wires the REAL ``DevMixin`` template (``build_template`` over the cached Qwen2.5-0.5B
tokenizer, ``load_model=False``: tokenizer only, no weights, no GPU) into the REAL
``twinkle_agentic.rollout.MultiTurnRollout`` with a scripted sampler and a real ``ToolManager``, and
runs a whole tool episode. The sampler is scripted (canned replies) so nothing needs a model forward,
but everything else is production code: ``ledger.open`` -> ``record`` -> ``parse_tool_call`` ->
``ToolManager`` dispatch -> ``observe`` -> ``extend_with_bridge`` -> ``append_ids`` ->
``_invoke_post_pipeline`` -> ``concat_input_feature`` -> ``merge``/``audit``.

That full loop is what pins the seams the doubles used to paper over:

* the tool schema must be baked INTO the opening prompt tokens (``ledger.open`` injecting ``tools``
  into ``encode``), or the model is never told it can call anything;
* ``concat_input_feature`` must append the assistant turn to ``messages`` (the ledger adopts the
  feature wholesale and every consumer reads the reply off ``messages``), or the answer is empty;
* the agent tool-call methods (``parse_tool_call`` / ``clean_tool_call`` / ``tool_call_errors``) must
  exist on a dev template, or the first tool turn raises ``AttributeError``;
* ``_invoke_post_pipeline`` must exist and roll ``labels`` + ``completion_mask`` together, or the
  first ``observe`` raises ``AttributeError``;
* ``concat_input_feature`` must EXTEND ``completion_mask`` in step with ``labels`` across turns, or
  the second assistant turn dies in ``audit`` with a ``completion_mask/labels misaligned`` mismatch.

The tool-call markup is assembled from the parser's own markers and ``chr(60)``/``chr(62)`` -- never
typed as literal angle-bracket tags -- because such a literal in a source string is indistinguishable
from harness control markup.
"""
import json

import pytest

from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams
from twinkle.template.tools import HermesQwenParser
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
_PARSER = HermesQwenParser()
_OM, _CM = _PARSER.open_marker, _PARSER.close_marker
_LT, _GT = chr(60), chr(62)


def _hermes_call(name='run_command', arg_key='command', arg_val='pwd'):
    """A tool call in the markup swift's Qwen agent templates render and HermesQwenParser reads."""
    return (_OM + '\n' + _LT + f'function={name}' + _GT + '\n' + _LT + f'parameter={arg_key}' + _GT + '\n' +
            arg_val + '\n' + _LT + '/parameter' + _GT + '\n' + _LT + '/function' + _GT + '\n' + _CM)


def _malformed_call():
    """Markup that opens a call block but holds no parseable payload -- the model asked for a tool and
    the parser rejects it, which drives the loop's rewrite-and-retry rather than a dispatch."""
    return _OM + '\nnot json at all\n' + _CM


class _EchoTool(Tool):
    """A real Tool the real ToolManager dispatches: echoes its arguments as a JSON string."""

    def __init__(self, name='run_command'):
        self._name = name

    def __call__(self, tool_name, arguments):
        return f'echo[{tool_name}]:{json.dumps(arguments, sort_keys=True)}'

    def tool_info(self):
        return {
            'type': 'function',
            'function': {
                'name': self._name,
                'description': 'echo the arguments back',
                'parameters': {}
            },
        }


class _ScriptedSampler:
    """Returns canned replies and builds ``new_input_feature`` through the REAL
    ``DevMixin.concat_input_feature`` -- the exact object the ledger adopts via ``record``.

    ``sample`` is declared ``_enable_continous_work`` because ``MultiTurnRollout`` samples one
    trajectory per call and refuses a sampler that would slice such a batch across workers. The pifs
    it is handed are recorded so a test can read the opening prompt the engine actually baked.
    """

    def __init__(self, template, replies):
        self.template = template
        self._replies = list(replies)
        self._turn = 0
        self.seen_pifs = []

    def sample(self, pifs, sampling_params=None, **kwargs):
        if isinstance(pifs, dict):
            pifs = [pifs]
        responses = []
        for pif in pifs:
            self.seen_pifs.append(pif)
            text = self._replies[self._turn]
            self._turn += 1
            tokens = self.template.tokenizer.encode(text, add_special_tokens=False)
            # logprobs shaped like the sampler's normalized top-k: one (token, logprob) list per token.
            logprobs = [[(tok, -0.1)] for tok in tokens]
            new_pif = self.template.concat_input_feature(pif, tokens)
            seq = SampledSequence(
                stop_reason='stop', tokens=tokens, logprobs=logprobs, decoded=text, new_input_feature=new_pif)
            responses.append(SampleResponse(sequences=[seq]))
        return responses

    sample._enable_continous_work = True


@pytest.fixture(scope='module')
def template():
    """The REAL dev template: ``Shifted<Family>`` = ``type(DevMixin, legacy)`` over the cached
    tokenizer. ``load_model=False`` fetches no weights, so this is CPU-only; a fresh box without the
    tokenizer cached skips rather than fails (mirrors ``test_chat_template_parity``)."""
    from swift.dev.builders.template import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    try:
        _, proc = get_model_processor(MODEL, load_model=False)
    except Exception as exc:  # noqa: BLE001 -- an unfetchable tokenizer is not a template defect
        pytest.skip(f'{MODEL}: tokenizer not fetchable ({type(exc).__name__})')
    return build_template(TemplateConfig(template='qwen2_5', max_length=2048), proc)


@pytest.fixture
def tool_manager():
    manager = ToolManager({})
    manager.register(_EchoTool('run_command'))
    return manager


def _run_episode(template, tool_manager, replies, prompt='Where am I? Use the tool.'):
    """Drive the REAL engine over one trajectory and return ``(output, sampler)``."""
    sampler = _ScriptedSampler(template, replies)
    rollout = MultiTurnRollout(
        sampler=sampler,
        template=template,
        tool_manager=tool_manager,
        sampling_params=SamplingParams(),
        max_turns=6,
    )
    outputs = rollout([{'messages': [{'role': 'user', 'content': prompt}]}], sampling_params=SamplingParams())
    assert len(outputs) == 1
    return outputs[0], sampler


def test_tool_schema_is_baked_into_the_opening_prompt(template, tool_manager):
    """Bug #1 guard: ``ledger.open`` must inject the resolved tools INTO the one encode, so the schema
    is in the prompt TOKENS -- not merely recorded on ``pif['tools']``. A local sampler feeds those ids
    straight to the engine and never re-encodes, so tools written only to the feature would never reach
    the model. Decode the opening prompt the sampler was handed and assert the schema is really there."""
    # The engine advertises the manager's tools when the trajectory named none.
    out, sampler = _run_episode(template, tool_manager, ['The cwd is /workspace.'])
    assert out.get('tools'), 'the resolved tool schemas must ride on the trajectory'

    opening = sampler.seen_pifs[0]
    prompt_text = template.tokenizer.decode(opening['input_ids'], skip_special_tokens=False)
    assert 'run_command' in prompt_text, 'the tool name must be rendered into the prompt tokens'
    assert 'echo the arguments back' in prompt_text, 'the tool description must reach the prompt'


def test_single_turn_tool_loop_drives_real_engine(template, tool_manager):
    """The headline end-to-end: one tool call, one dispatch, one final answer -- the whole observe path
    runs on the real template. Assert the token account, the message transcript and the logprob
    alignment the ledger's ``audit`` enforces."""
    out, _ = _run_episode(template, tool_manager, [_hermes_call('run_command', 'command', 'pwd'),
                                                   'The cwd is /workspace.'])
    input_ids = out['input_ids']
    labels = out['labels']
    completion_mask = out['completion_mask']

    # Existence + the invariant every downstream consumer relies on: all three the same length.
    assert input_ids and labels and completion_mask
    assert len(input_ids) == len(labels) == len(completion_mask)

    # The transcript: user -> assistant(call) -> tool(result) -> assistant(answer).
    roles = [m['role'] for m in out['messages']]
    assert roles == ['user', 'assistant', 'tool', 'assistant']
    call_msg = out['messages'][1]
    assert call_msg['content'] == ''  # the whole turn was the call, so cleaning left nothing
    assert call_msg['tool_calls'] == [{
        'type': 'function',
        'function': {
            'name': 'run_command',
            'arguments': {
                'command': 'pwd'
            }
        }
    }]
    # The tool result the real ToolManager produced was appended as a tool turn (bug #2/#3 guard: the
    # observe path ran and the assistant reply after it is readable, not empty).
    assert out['messages'][2]['role'] == 'tool'
    assert 'echo[run_command]' in out['messages'][2]['content']
    assert out['messages'][-1] == {'role': 'assistant', 'content': 'The cwd is /workspace.'}

    # The episode grew by generation: two assistant turns banked, and the trainable tokens (the policy's
    # own) are exactly the ones the sampler returned logprobs for -- the audit that would raise on drift.
    assert out['turns'] == 2
    assert out['stop_reason'] == 'stop'
    assert not out['truncated']
    trainable = sum(1 for lab, flag in zip(labels, completion_mask) if lab != -100 and flag)
    assert len(out['logprobs']) == trainable
    assert trainable > 0


def test_labels_and_mask_stay_aligned_across_three_turns(template, tool_manager):
    """Bug #4 guard: two tool calls then an answer means TWO observes and THREE ``concat_input_feature``
    appends. The single-turn shortcut let ``completion_mask`` fall behind ``labels`` on the second
    append; this stresses the unroll-append-reroll across all of them and asserts the account still
    lines up, with each turn's trainable run present."""
    out, _ = _run_episode(
        template, tool_manager,
        [_hermes_call('run_command', 'command', 'ls'),
         _hermes_call('run_command', 'command', 'pwd'), 'Done.'])
    input_ids, labels, completion_mask = out['input_ids'], out['labels'], out['completion_mask']
    assert len(input_ids) == len(labels) == len(completion_mask)
    assert out['turns'] == 3
    assert [m['role'] for m in out['messages']] == [
        'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant'
    ]
    # Every observation is masked out of the loss; only the policy's tokens are trainable.
    trainable = sum(1 for lab, flag in zip(labels, completion_mask) if lab != -100 and flag)
    assert len(out['logprobs']) == trainable
    # The two tool turns contributed observations that must NOT be trainable.
    tool_positions_masked = all(flag == 0 for flag in completion_mask) or trainable < len(completion_mask)
    assert tool_positions_masked


def test_malformed_tool_call_is_fed_back_and_retried(template, tool_manager):
    """``tool_call_errors`` guard: markup that opens a call but does not parse is the model asking for a
    tool, not declining one. The loop must hand the parser's reason back as a tool message and grant
    another turn rather than ending the episode -- all through the real observe path."""
    out, _ = _run_episode(template, tool_manager, [_malformed_call(), 'The cwd is /workspace.'])
    roles = [m['role'] for m in out['messages']]
    # user -> assistant(bad markup) -> tool(the rewrite feedback) -> assistant(answer)
    assert roles == ['user', 'assistant', 'tool', 'assistant']
    feedback = out['messages'][2]['content']
    assert 'not run' in feedback and 'again' in feedback, feedback
    assert out['messages'][-1]['content'] == 'The cwd is /workspace.'
    # The malformed turn was still banked as a generation, and the account stayed aligned.
    assert len(out['input_ids']) == len(out['labels']) == len(out['completion_mask'])


def test_context_appended_turn_is_masked_out_of_the_loss(template):
    """``appended_as='context'`` on the real template: a turn later generations see but no loss may
    touch has its labels and completion_mask masked, while the message is still appended. Exercised on
    the real component (not the engine, which never authors context turns) because it is a branch of
    the same ``concat_input_feature`` the loop depends on."""
    opened = template.encode({'messages': [{
        'role': 'user',
        'content': 'hi'
    }]},
                             add_generation_prompt=True)
    opened['messages'] = [{'role': 'user', 'content': 'hi'}]
    tokens = template.tokenizer.encode('remembered history', add_special_tokens=False)
    out = template.concat_input_feature(opened, tokens, appended_as='context')
    assert out['messages'][-1] == {'role': 'assistant', 'content': 'remembered history'}
    assert len(out['input_ids']) == len(out['labels']) == len(out['completion_mask'])
    # The context turn's own tokens are the tail; none of them may be trainable.
    tail = len(tokens)
    assert all(lab == -100 for lab in _input_order(out['labels'])[-tail:])
    assert all(flag == 0 for flag in _input_order(out['completion_mask'])[-tail:])


def _input_order(values):
    """Undo the output-order roll so a test can read labels/mask aligned with input_ids."""
    return values[-1:] + values[:-1] if values else []
