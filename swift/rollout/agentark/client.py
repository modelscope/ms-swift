# Copyright (c) ModelScope Contributors. All rights reserved.
"""Asynchronous, dependency-light client for the AgentArk HTTP protocol."""

from __future__ import annotations

import aiohttp
import asyncio
import json
import os
import socket
import threading
import uuid
from copy import deepcopy
from typing import Any, Mapping

from .heartbeat import LeaseHandle


class AgentArkHttpError(RuntimeError):
    """An AgentArk request failed or returned an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        path: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        code: str | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.path = path
        self.status_code = status_code
        self.response_body = response_body
        self.code = code
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class AgentArkStaleLeaseError(AgentArkHttpError):
    """The server rejected a request using an obsolete lease identity."""


_CLIENT_ID_LOCK = threading.Lock()
_CLIENT_ID_PID: int | None = None
_CLIENT_ID: str | None = None
_OPERATION_NAMESPACE = uuid.UUID('6770be85-41ef-4dc0-a1a4-df9f3b21f84d')


def process_client_id() -> str:
    """Return a process-local identity and regenerate it after ``fork``."""

    configured = str(os.getenv('AGENTARK_CLIENT_ID') or '').strip()
    if configured:
        return configured
    global _CLIENT_ID, _CLIENT_ID_PID
    pid = os.getpid()
    with _CLIENT_ID_LOCK:
        if _CLIENT_ID is None or _CLIENT_ID_PID != pid:
            host = socket.gethostname().split('.', 1)[0] or 'host'
            _CLIENT_ID = f'swift-{host}-{pid}-{uuid.uuid4().hex[:12]}'
            _CLIENT_ID_PID = pid
        return _CLIENT_ID


def stable_operation_id(kind: str, *parts: str) -> str:
    """Build a compact deterministic idempotency id from stable inputs."""

    name = '\x1f'.join([str(kind), *(str(part) for part in parts)])
    return f'{kind}-{uuid.uuid5(_OPERATION_NAMESPACE, name).hex}'


def _error_detail(payload: Any) -> tuple[str | None, bool | None, str | None]:
    if not isinstance(payload, Mapping):
        return None, None, None
    detail: Any = payload.get('error')
    if not isinstance(detail, Mapping):
        detail = payload.get('detail')
    if isinstance(detail, Mapping) and isinstance(detail.get('error'), Mapping):
        detail = detail['error']
    if not isinstance(detail, Mapping):
        # Some endpoints may return the stable error shape at top level.
        detail = payload if 'code' in payload else None
    if not isinstance(detail, Mapping):
        return None, None, None
    code_value = detail.get('code')
    code = None if code_value in (None, '') else str(code_value)
    retryable_value = detail.get('retryable')
    retryable = None if retryable_value is None else bool(retryable_value)
    message_value = detail.get('message')
    message = None if message_value in (None, '') else str(message_value)
    return code, retryable, message


def _is_stale_lease_conflict(
    *,
    status_code: int | None,
    code: str | None,
    message: str | None,
    response_body: str | None,
) -> bool:
    """Recognize ownership conflicts without treating every HTTP 409 as stale.

    A 409 such as ``operation_in_progress`` or a semantic turn conflict is
    handled by the normal HTTP client policy.  The server versions used by
    AgentArk have emitted both structured and plain-text stale-ownership
    messages, so keep the fallback deliberately narrow.
    """

    if status_code not in {409, 410}:
        return False
    normalized_code = (code or '').strip().lower()
    if normalized_code in {
            'lease_gone',
            'stale_lease',
            'lease_stale',
            'lease_identity_conflict',
            'lease_generation_conflict',
            'server_epoch_conflict',
    }:
        return True
    text = ' '.join(part.strip().lower() for part in (message or '', response_body or '') if part)
    return ('lease identity does not own env_id' in text or 'belongs to an older generation' in text
            or 'older lease generation' in text or 'server epoch' in text and 'lease' in text)


def _http_error(
    *,
    message: str,
    path: str,
    status_code: int | None,
    response_body: str | None,
    code: str | None = None,
    retryable: bool | None = None,
    retry_after_s: float | None = None,
) -> AgentArkHttpError:
    error_type = (
        AgentArkStaleLeaseError if _is_stale_lease_conflict(
            status_code=status_code,
            code=code,
            message=message,
            response_body=response_body,
        ) else AgentArkHttpError)
    return error_type(
        message,
        method='POST',
        path=path,
        status_code=status_code,
        response_body=response_body,
        code=code,
        retryable=retryable,
        retry_after_s=retry_after_s,
    )


class AgentArkHttpClient:
    """Async client for v2 leases with an explicit v1 compatibility mode.

    V2 acquire, step, and release carry idempotency IDs, so transport failures,
    HTTP 5xx responses, and ``operation_in_progress`` may be retried with the
    exact same payload.  V1 keeps its original no-retry behavior because a lost
    response could otherwise acquire twice or execute an action twice.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 600.0,
        release_timeout_s: float = 30.0,
        protocol_version: str | None = None,
        client_id: str | None = None,
        v2_max_attempts: int = 3,
        v2_retry_base_delay_s: float = 0.1,
        v2_retry_max_delay_s: float = 1.0,
        client: aiohttp.ClientSession | None = None,
    ) -> None:
        base_url = str(base_url or '').strip().rstrip('/')
        if not base_url:
            raise ValueError('AgentArk server base_url must not be empty')
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        if release_timeout_s <= 0:
            raise ValueError('release_timeout_s must be positive')
        resolved_protocol = str(protocol_version or os.getenv('AGENTARK_PROTOCOL_VERSION') or 'v2').strip().lower()
        if resolved_protocol not in {'v1', 'v2'}:
            raise ValueError("protocol_version must be 'v1' or 'v2'")
        if int(v2_max_attempts) <= 0:
            raise ValueError('v2_max_attempts must be positive')
        if float(v2_retry_base_delay_s) < 0 or float(v2_retry_max_delay_s) < 0:
            raise ValueError('v2 retry delays must not be negative')

        self.base_url = base_url
        self.timeout_s = float(timeout_s)
        self.release_timeout_s = float(release_timeout_s)
        self.protocol_version = resolved_protocol
        self._explicit_client_id = None if client_id in (None, '') else str(client_id)
        self.v2_max_attempts = int(v2_max_attempts)
        self.v2_retry_base_delay_s = float(v2_retry_base_delay_s)
        self.v2_retry_max_delay_s = float(v2_retry_max_delay_s)
        # Swift colocate may drive reset, turn-end, and cleanup hooks through
        # separate short-lived asyncio loops. A pooled
        # ClientSession opened during reset retains transports bound to that
        # first loop and fails later with ``Event loop is closed``. The normal
        # path therefore creates one client per request in the *current* loop.
        # An explicitly injected client is retained for transport-level tests;
        # its lifecycle and loop affinity remain the caller's responsibility.
        self._client = client
        self._closed = False

    @property
    def client_id(self) -> str:
        return self._explicit_client_id or process_client_id()

    async def _post_once(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError('AgentArkHttpClient is closed')
        try:
            if self._client is None:
                timeout = aiohttp.ClientTimeout(total=timeout_s)
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as request_client:
                    async with request_client.post(f'{self.base_url}{path}', json=dict(payload)) as response:
                        status_code = response.status
                        headers = response.headers
                        response_bytes = await response.read()
            else:
                timeout = aiohttp.ClientTimeout(total=timeout_s)
                async with self._client.post(f'{self.base_url}{path}', json=dict(payload), timeout=timeout) as response:
                    status_code = response.status
                    headers = response.headers
                    response_bytes = await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise AgentArkHttpError(
                f'AgentArk POST {path} failed: {type(exc).__name__}: {exc}',
                method='POST',
                path=path,
            ) from exc

        body = response_bytes.decode('utf-8', errors='replace')[:4000]
        try:
            result = json.loads(response_bytes)
        except (TypeError, ValueError) as exc:
            if status_code >= 400:
                raise _http_error(
                    message=f'AgentArk POST {path} returned HTTP {status_code}: {body}',
                    path=path,
                    status_code=status_code,
                    response_body=body,
                    retry_after_s=self._retry_after(headers),
                ) from exc
            raise AgentArkHttpError(
                f'AgentArk POST {path} returned non-JSON data: {body}',
                method='POST',
                path=path,
                status_code=status_code,
                response_body=body,
            ) from exc

        code, retryable, detail_message = _error_detail(result)
        if status_code >= 400:
            suffix = detail_message or body
            raise _http_error(
                message=f'AgentArk POST {path} returned HTTP {status_code}: {suffix}',
                path=path,
                status_code=status_code,
                response_body=body,
                code=code,
                retryable=retryable,
                retry_after_s=self._retry_after(headers),
            )
        if not isinstance(result, dict):
            raise AgentArkHttpError(
                f'AgentArk POST {path} returned {type(result).__name__}, expected an object',
                method='POST',
                path=path,
                status_code=status_code,
                response_body=body,
            )
        if code == 'operation_in_progress':
            raise AgentArkHttpError(
                f'AgentArk POST {path} reported operation_in_progress: {detail_message or body}',
                method='POST',
                path=path,
                status_code=status_code,
                response_body=body,
                code=code,
                retryable=True,
                retry_after_s=self._retry_after(headers),
            )
        return result

    async def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
        allow_v2_retry: bool = False,
    ) -> dict[str, Any]:
        request_timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        stable_payload = deepcopy(dict(payload))
        attempts = self.v2_max_attempts if allow_v2_retry else 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._post_once(path, stable_payload, timeout_s=request_timeout)
            except AgentArkHttpError as exc:
                should_retry = (
                    allow_v2_retry and attempt < attempts and
                    (exc.status_code is None or
                     (exc.status_code is not None and exc.status_code >= 500) or exc.code == 'operation_in_progress'))
                if not should_retry:
                    raise
                delay = min(
                    self.v2_retry_max_delay_s,
                    self.v2_retry_base_delay_s * (2**(attempt - 1)),
                )
                if exc.retry_after_s is not None:
                    delay = min(self.v2_retry_max_delay_s, max(delay, exc.retry_after_s))
                if delay > 0:
                    await asyncio.sleep(delay)
        raise AssertionError('unreachable')

    @staticmethod
    def _retry_after(headers: Mapping[str, str]) -> float | None:
        value = headers.get('Retry-After')
        if value in (None, ''):
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    async def acquire_start(
        self,
        env_cfg: Mapping[str, Any],
        *,
        uid: str,
        task_name: str | None = None,
        group_seed: int | None = None,
        env_id: str | None = None,
        unity_env_id: int | None = None,
        acquire_request_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Lease a runtime and reset it in one server operation."""

        payload = {
            'cfg': deepcopy(dict(env_cfg)),
            'env_id': env_id,
            'task_name': task_name,
            'group_seed': group_seed,
            'unity_env_id': unity_env_id,
            'uid': uid,
        }
        if self.protocol_version == 'v1':
            return await self._post('/v1/envs/acquire_start', payload)
        request_id = str(acquire_request_id or '').strip()
        if not request_id:
            raise ValueError('acquire_request_id is required by AgentArk protocol v2')
        payload = {
            'acquire_request_id': request_id,
            'client_id': str(client_id or self.client_id),
            **payload,
        }
        return await self._post('/v2/envs/acquire_start', payload, allow_v2_retry=True)

    async def step(
        self,
        lease: LeaseHandle | str,
        *,
        action: str | None,
        assistant: str,
        action_id: str | None = None,
        turn_index: int | None = None,
    ) -> dict[str, Any]:
        """Execute one model action, replay-safe under protocol v2."""

        if self.protocol_version == 'v1':
            env_id = lease.env_id if isinstance(lease, LeaseHandle) else str(lease)
            return await self._post(
                f'/v1/envs/{env_id}/step',
                {
                    'action': action,
                    'assistant': assistant
                },
            )
        if not isinstance(lease, LeaseHandle):
            raise TypeError('AgentArk protocol v2 step requires a LeaseHandle')
        resolved_action_id = str(action_id or '').strip()
        if not resolved_action_id:
            raise ValueError('action_id is required by AgentArk protocol v2')
        if turn_index is None or int(turn_index) <= 0:
            raise ValueError('turn_index must be positive for AgentArk protocol v2')
        lease.assert_active()
        payload = {
            **lease.request_identity(),
            'action_id': resolved_action_id,
            'turn_index': int(turn_index),
            'action': action,
            'assistant': assistant,
        }
        try:
            result = await self._post(
                f'/v2/envs/{lease.env_id}/step',
                payload,
                allow_v2_retry=True,
            )
        except AgentArkStaleLeaseError as exc:
            lease.mark_expired(f'server rejected stale lease identity: {exc}')
            raise
        if bool(result.get('lease_expired_after_step')):
            lease.mark_expired('server expired the lease while this step was in progress')
        else:
            lease.mark_alive(
                lease_ttl_s=self._optional_number(result.get('lease_ttl_s')),
                lease_expires_in_s=self._optional_number(result.get('lease_expires_in_s')),
            )
        return result

    async def release(
        self,
        lease: LeaseHandle | str,
        *,
        release_request_id: str | None = None,
    ) -> dict[str, Any]:
        """Return one exact leased runtime to AgentArk's pool."""

        if self.protocol_version == 'v1':
            env_id = lease.env_id if isinstance(lease, LeaseHandle) else str(lease)
            return await self._post(
                f'/v1/envs/{env_id}/release',
                {},
                timeout_s=self.release_timeout_s,
            )
        if not isinstance(lease, LeaseHandle):
            raise TypeError('AgentArk protocol v2 release requires a LeaseHandle')
        request_id = str(release_request_id or lease.release_request_id).strip()
        if not request_id:
            raise ValueError('release_request_id is required by AgentArk protocol v2')
        try:
            return await self._post(
                f'/v2/envs/{lease.env_id}/release',
                {
                    **lease.request_identity(), 'release_request_id': request_id
                },
                timeout_s=self.release_timeout_s,
                allow_v2_retry=True,
            )
        except AgentArkStaleLeaseError:
            # The server no longer recognizes this identity.  From the
            # client's perspective the lease is already gone; do not turn
            # cleanup of an obsolete lease into a second failure.
            lease.mark_expired('server rejected stale lease during release')
            return {'ok': True, 'stale_lease': True}

    @staticmethod
    def _optional_number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

    async def __aenter__(self) -> 'AgentArkHttpClient':
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()
