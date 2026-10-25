# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shutdown / restart lifecycle of ``swift deploy`` (requirement 5).

End-to-end on the real stack: bring a server up on a fixed port, serve a chat completion, tear it down
(leaving the ``live_server`` context manager), assert the process really released the port, then bring a
*new* server up on the **same** port and assert it serves again. Rebinding the identical port is the part
that would silently break if teardown left a stray Ray Serve proxy or a lingering socket behind, so this
test is the guard that a deploy can be stopped and restarted in place.
"""
from __future__ import annotations

import time

import pytest

from swift.dev.tests.deploy.conftest import SERVED_NAME, free_port, port_is_closed

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: A chat turn reused for every liveness probe; short ``max_tokens`` keeps each generation cheap.
_MESSAGES = [{'role': 'user', 'content': 'Reply with the single word: ok'}]


def _wait_port_closed(port: int, timeout: float = 90.0) -> bool:
    """Poll until nothing accepts connections on ``port`` (the torn-down server let go of it)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_closed(port):
            return True
        time.sleep(1.0)
    return port_is_closed(port)


def _assert_serves(srv) -> None:
    """A torn-down-then-restarted (or first-launch) server must list the model and complete a chat."""
    models = srv.get('/models')
    assert models.status_code == 200
    assert SERVED_NAME in [m['id'] for m in models.json()['data']]
    body = srv.chat(_MESSAGES, max_tokens=8)
    assert body['object'] == 'chat.completion'
    assert body['model'] == SERVED_NAME
    assert body['choices'][0]['message']['content']
    assert body['usage']['completion_tokens'] > 0


def test_server_shuts_down_and_restarts_on_the_same_port(live_server):
    port = free_port()

    # First launch: the server is reachable and answers a chat completion.
    with live_server(backend='vllm', port=port) as first:
        assert first.port == port
        _assert_serves(first)

    # Teardown ran on context exit: the process must release the port before anything can rebind it.
    assert _wait_port_closed(port), f'port {port} still accepting connections after teardown'

    # Restart on the identical port: a fresh server binds it and serves the same way.
    with live_server(backend='vllm', port=port) as second:
        assert second.port == port
        _assert_serves(second)

    # And it too releases the port on shutdown.
    assert _wait_port_closed(port), f'port {port} still accepting connections after second teardown'
