# Copyright (c) ModelScope Contributors. All rights reserved.
"""Thin wrapper around OpenEnv's GenericEnvClient with reconnection support.

This wrapper only manages the WebSocket connection and forwards raw actions
to the server.  Action parsing (LLM text → dict) and observation formatting
(dict → LLM string) are handled by the :class:`OpenEnvScheduler` subclass,
not here.

env_config keys consumed:
    base_url (str): OpenEnv server URL (e.g. "http://localhost:8000").
    reset_kwargs (dict, optional): Extra kwargs passed to client.reset().
"""
import time
from typing import Any, Dict, Tuple

from swift.utils import get_logger

logger = get_logger()

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


class OpenEnvWrapper:
    """Thin wrapper around ``GenericEnvClient`` with reconnection support.

    Unlike the previous version, this class does **not** inherit from
    ``Env`` and does **not** parse LLM text or format observations.
    All such logic lives in :class:`~swift.rollout.multi_turn.OpenEnvScheduler`
    and its subclasses.
    """

    def __init__(self, env_config: Dict[str, Any]):
        self.base_url = env_config.get('base_url', 'http://localhost:8000')
        self.reset_kwargs = env_config.get('reset_kwargs', {})
        self._client = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_client(self):
        """Lazily create the OpenEnv client (sync wrapper)."""
        if self._client is None:
            from openenv.core.generic_client import GenericEnvClient

            client = GenericEnvClient(base_url=self.base_url)
            sync_client = client.sync()
            sync_client.__enter__()
            self._client = sync_client
        return self._client

    def _reconnect_client(self):
        """Close old client (recover from connection loss).

        Only cleans up; the next _ensure_client() call will create a new connection.
        """
        if self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception:
                pass
            self._client = None

    def _call_with_retry(self, fn_name: str, *args, **kwargs):
        """Call a client method with retry on connection failure."""
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                client = self._ensure_client()
                return getattr(client, fn_name)(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(f"OpenEnv {fn_name} failed (attempt {attempt + 1}/{_MAX_RETRIES}): {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (attempt + 1))
                    self._reconnect_client()
        raise last_exc

    # ------------------------------------------------------------------
    # Public API (synchronous — called from async scheduler via run_in_executor
    # or directly since GenericEnvClient.sync() is already blocking)
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Any, Any]:
        """Reset the OpenEnv environment.

        Returns:
            (raw_observation, metadata) — unformatted, as returned by the server.
        """
        result = self._call_with_retry('reset', **self.reset_kwargs)
        return result.observation, getattr(result, 'metadata', None)

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Any, float, bool, Any]:
        """Execute one step in the OpenEnv environment.

        Args:
            action_dict: Pre-parsed action dict (e.g. ``{"answer": "7"}``).

        Returns:
            (raw_observation, reward, done, metadata) — unformatted.
        """
        result = self._call_with_retry('step', action_dict)
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        return result.observation, reward, done, getattr(result, 'metadata', None)

    def close(self):
        """Close the OpenEnv client connection."""
        if self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception:
                logger.debug('OpenEnv client close failed', exc_info=True)
            self._client = None
