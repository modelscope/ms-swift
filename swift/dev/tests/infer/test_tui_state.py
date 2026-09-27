# Copyright (c) ModelScope Contributors. All rights reserved.
"""Interactive REPL state-machine tests (``infer_tui._CliState``), no model, no GPU.

``_CliState`` is the pure conversation state behind ``infer_cli`` (legacy's ``InferCliState``): history
building, the user-boundary prune, the trajectory it hands the sampler (system + media), and the
in-REPL commands. ``input`` is monkeypatched so the command handling is exercised without a terminal.
"""
import pytest

from swift.dev.recipe.infer_tui import _MAX_HISTORY_MESSAGES, _QUIT, _CliState


def _feed(monkeypatch, lines):
    """Make ``input()`` return successive lines, recording how many were consumed."""
    state = {'i': 0}

    def fake_input(prompt=''):
        if state['i'] >= len(lines):
            raise EOFError
        value = lines[state['i']]
        state['i'] += 1
        return value

    monkeypatch.setattr('builtins.input', fake_input)
    return state


def test_add_query_and_response_build_alternating_history():
    state = _CliState()
    state.add_query('hi')
    state.add_response('hello')
    assert state.messages == [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]


def test_to_trajectory_prepends_system_and_attaches_media():
    state = _CliState(system='be brief')
    state.add_query('hi')
    state.media['images'] = ['/tmp/a.png']
    trajectory = state.to_trajectory()
    assert trajectory['messages'][0] == {'role': 'system', 'content': 'be brief'}
    assert trajectory['messages'][1] == {'role': 'user', 'content': 'hi'}
    assert trajectory['images'] == ['/tmp/a.png']
    # empty media kinds are not attached
    assert 'audios' not in trajectory and 'videos' not in trajectory


def test_to_trajectory_without_system_has_no_system_turn():
    state = _CliState()
    state.add_query('hi')
    assert state.to_trajectory()['messages'][0]['role'] == 'user'


def test_clear_resets_messages_and_media():
    state = _CliState()
    state.add_query('hi')
    state.media['images'] = ['/tmp/a.png']
    state.clear()
    assert state.messages == []
    assert state.media == {'images': [], 'audios': [], 'videos': []}


def test_prune_cuts_at_a_user_boundary():
    """History is bounded, and the cut lands on a user turn so an assistant answer is never orphaned."""
    state = _CliState()
    # 60 alternating messages starting with a user turn (even index = user)
    for i in range(30):
        state.messages.append({'role': 'user', 'content': f'q{i}'})
        state.messages.append({'role': 'assistant', 'content': f'a{i}'})
    assert len(state.messages) == 60
    state._prune()
    assert len(state.messages) <= _MAX_HISTORY_MESSAGES
    assert state.messages[0]['role'] == 'user'


def test_prune_noop_under_the_bound():
    state = _CliState()
    state.add_query('hi')
    state.add_response('hello')
    state._prune()
    assert len(state.messages) == 2


def test_read_query_returns_plain_text_stripped(monkeypatch):
    _feed(monkeypatch, ['  hello there  '])
    state = _CliState()
    assert state.read_query() == 'hello there'


def test_read_query_empty_returns_none(monkeypatch):
    _feed(monkeypatch, ['   '])
    assert _CliState().read_query() is None


@pytest.mark.parametrize('cmd', ['quit', 'exit', 'q', 'QUIT', 'Exit'])
def test_read_query_quit_commands(monkeypatch, cmd):
    _feed(monkeypatch, [cmd])
    assert _CliState().read_query() is _QUIT


def test_read_query_clear_command(monkeypatch):
    _feed(monkeypatch, ['clear'])
    state = _CliState()
    state.add_query('old')
    assert state.read_query() is None  # handled, not a query
    assert state.messages == []


def test_read_query_reset_system_command(monkeypatch):
    # 'reset-system' then the new system prompt on the next input()
    _feed(monkeypatch, ['reset-system', 'you are terse'])
    state = _CliState(system='old')
    state.add_query('history')
    assert state.read_query() is None
    assert state.system == 'you are terse'
    assert state.messages == []  # history cleared on a system swap


def test_read_query_multiline_toggle(monkeypatch):
    _feed(monkeypatch, ['multi-line'])
    state = _CliState()
    assert state.read_query() is None
    assert state.multiline is True
    # in multiline mode _read_raw collects until a lone '#', and the joined text is still command-checked
    _feed(monkeypatch, ['single-line', '#'])
    assert state.read_query() is None
    assert state.multiline is False


def test_read_query_multiline_collects_until_hash(monkeypatch):
    _feed(monkeypatch, ['multi-line'])
    state = _CliState()
    state.read_query()  # turn multiline on
    _feed(monkeypatch, ['line one', 'line two', '#'])
    assert state.read_query() == 'line one\nline two'


def test_prompt_media_collects_until_blank(monkeypatch):
    _feed(monkeypatch, ['/tmp/a.png', '/tmp/b.png', '', ''])
    state = _CliState()
    state.prompt_media(['images', 'audios'])
    assert state.media['images'] == ['/tmp/a.png', '/tmp/b.png']
    assert state.media['audios'] == []  # blank immediately -> none
