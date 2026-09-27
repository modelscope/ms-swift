# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end interactive REPL (``infer_cli``) coverage with a real backend + TinyModel.

``test_tui_state.py`` (fast tier) unit-tests ``_CliState`` -- command parsing, history pruning, the
multi-line toggle -- with a stubbed sampler and no model. This tier drives the WHOLE REPL loop through
``infer_cli``: build the template + a real transformers sampler off a ``TinyModel``, read a turn from a
scripted stdin, sample a real completion, echo it, keep history, and shut the sampler down on ``quit``.
The weights are random so the reply text is meaningless; the assertions are on the loop's observable
contract (one ``Agent:`` echo per query, history carried across turns, commands handled, a clean
``Bye.``), not on what the model said.

``infer_cli`` builds its own sampler and initialises the backend internally, so -- unlike ``run_infer``'s
generative path -- the test needs no explicit twinkle init. ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``.
"""
import builtins

import pytest

from swift.dev.tests.tiny import TinyModel

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    """Build the 4-layer TinyModel once for the whole module (each REPL call reloads a sampler off it)."""
    dest = tmp_path_factory.mktemp('tui_model')
    return TinyModel.build(dest / 'model')


def _script_stdin(monkeypatch, lines):
    """Feed ``lines`` to ``input()`` in order, then raise EOFError (which the REPL treats as a clean quit)."""
    state = {'i': 0}

    def fake_input(prompt=''):
        i = state['i']
        state['i'] += 1
        if i < len(lines):
            return lines[i]
        raise EOFError

    monkeypatch.setattr(builtins, 'input', fake_input)


def _run_repl(tiny_model, monkeypatch, capsys, lines, *, multi_round=True):
    from swift.dev.config import GenerationConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.infer_tui import infer_cli

    _script_stdin(monkeypatch, lines)
    infer_cli(
        ModelConfig(
            model=tiny_model, model_type=TinyModel.MODEL_TYPE, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template=TinyModel.TEMPLATE, max_length=256),
        GenerationConfig(max_new_tokens=8, temperature=0.8),
        backend='transformers',
        multi_round=multi_round)
    return capsys.readouterr().out


def test_single_turn_echoes_a_response_and_quits_cleanly(tiny_model, monkeypatch, capsys):
    """One query -> one real sampled reply echoed as ``Agent:``, then ``quit`` shuts the sampler down."""
    out = _run_repl(tiny_model, monkeypatch, capsys, ['Hello there', 'quit'])

    assert out.count('Agent:') == 1, f'expected exactly one agent turn, got:\n{out}'
    assert 'Bye' in out, 'quit did not reach the clean shutdown message'


def test_multi_round_runs_one_turn_per_query(tiny_model, monkeypatch, capsys):
    """With ``multi_round`` on, every non-command query gets its own turn and history is carried forward."""
    out = _run_repl(tiny_model, monkeypatch, capsys, ['First question', 'Second question', 'quit'])

    assert out.count('Agent:') == 2, f'expected two agent turns, got:\n{out}'
    assert 'Bye' in out


def test_clear_command_resets_history_mid_session(tiny_model, monkeypatch, capsys):
    """``clear`` is handled as a command (no turn), prints its confirmation, and the session continues."""
    out = _run_repl(tiny_model, monkeypatch, capsys, ['Hi', 'clear', 'Again', 'quit'])

    assert 'History cleared' in out, f'clear command was not handled:\n{out}'
    # two real queries ('Hi', 'Again') -> two turns; 'clear' produced none
    assert out.count('Agent:') == 2, f'expected two agent turns around the clear, got:\n{out}'
    assert 'Bye' in out


def test_eof_without_quit_still_shuts_down(tiny_model, monkeypatch, capsys):
    """Running out of stdin (EOFError) is a clean exit too -- the sampler is shut down in the finally."""
    out = _run_repl(tiny_model, monkeypatch, capsys, ['Only one turn'])

    assert out.count('Agent:') == 1
    assert 'Bye' in out, 'an EOF exit skipped the shutdown path'
