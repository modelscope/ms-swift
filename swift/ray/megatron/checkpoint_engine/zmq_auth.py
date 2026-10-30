# Copyright (c) ModelScope Contributors. All rights reserved.
"""HMAC authentication for ZMQ control-channel messages.

Provides ``send_auth_pyobj`` / ``recv_auth_pyobj`` helpers that prepend an
HMAC-SHA256 signature to pickled payloads, so that a crafted pickle from an
untrusted peer can never reach the deserializer.

When ``auth_key`` is empty the helpers fall back to plain ``pickle.dumps`` /
``pickle.loads`` — i.e. the original, unauthenticated behaviour — so callers
that have not opted into authentication are unaffected.
"""

from __future__ import annotations

import hashlib
import hmac
import pickle
import secrets
import zmq
from typing import Any

_HMAC_SIZE = 32  # SHA-256 digest size in bytes


def generate_auth_key() -> str:
    """Generate a 256-bit shared secret as a hex string."""
    return secrets.token_hex(32)


def send_auth_pyobj(
    socket: zmq.Socket,
    payload: Any,
    auth_key: str = '',
    *,
    where: str = '',
    timeout_s: float | None = None,
) -> None:
    """Send a pickled object, optionally with an HMAC signature prepended.

    When ``auth_key`` is set, the wire format is ``HMAC-SHA256(key, data) || data``.
    When empty, plain ``pickle.dumps(payload)`` is sent (backward compatible).
    """
    data = pickle.dumps(payload)
    if auth_key:
        sig = hmac.new(auth_key.encode(), data, hashlib.sha256).digest()
        msg = sig + data
    else:
        msg = data
    try:
        socket.send(msg)
    except zmq.error.Again as e:
        ts = f'{timeout_s}s' if timeout_s is not None else 'unknown'
        raise RuntimeError(f'ZMQ metadata timeout ({ts}) sending at {where}.') from e


def recv_auth_pyobj(
    socket: zmq.Socket,
    auth_key: str = '',
    *,
    where: str = '',
    timeout_s: float | None = None,
) -> Any:
    """Receive a pickled object after verifying its HMAC signature (if configured).

    The raw bytes are received first; the HMAC is checked *before*
    ``pickle.loads`` runs, so a crafted pickle from an untrusted peer can
    never reach the deserializer.

    When ``auth_key`` is empty, falls back to plain ``pickle.loads`` (backward
    compatible with the original ``recv_pyobj`` behaviour).
    """
    try:
        msg = socket.recv(copy=True)
    except zmq.error.Again as e:
        ts = f'{timeout_s}s' if timeout_s is not None else 'unknown'
        raise RuntimeError(f'ZMQ metadata timeout ({ts}) waiting at {where}.') from e

    if not auth_key:
        # No key configured — fall back to original recv_pyobj behaviour.
        return pickle.loads(msg) if isinstance(msg, (bytes, bytearray)) else socket.recv_pyobj()

    if len(msg) < _HMAC_SIZE:
        raise RuntimeError(f'Rejecting short ZMQ message at {where}: {len(msg)} bytes '
                           f'(< HMAC size {_HMAC_SIZE}).')

    received_hmac = msg[:_HMAC_SIZE]
    payload_bytes = msg[_HMAC_SIZE:]
    expected_hmac = hmac.new(auth_key.encode(), payload_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(received_hmac, expected_hmac):
        raise RuntimeError(f'HMAC verification failed for ZMQ message at {where}; '
                           'refusing to deserialize untrusted payload.')

    return pickle.loads(payload_bytes)
