# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process-wide lease heartbeats for AgentArk protocol v2.

Swift's colocate rollout driver enters several short-lived asyncio event loops
during one trajectory.  Heartbeats therefore live in one synchronous daemon
thread instead of being attached to any of those loops.
"""

from __future__ import annotations

import logging
import os
import requests
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class LeaseExpiredError(RuntimeError):
    """The server lease is no longer safe to use for another action."""


@dataclass
class LeaseHandle:
    """Capability and liveness state for one exact protocol-v2 lease."""

    server_epoch: str
    env_id: str
    lease_id: str
    lease_generation: int
    lease_ttl_s: float
    heartbeat_interval_s: float | None
    client_id: str
    acquire_request_id: str
    release_request_id: str
    owner_pid: int = field(default_factory=os.getpid)
    _expired: bool = field(default=False, init=False, repr=False, compare=False)
    _expired_reason: str | None = field(default=None, init=False, repr=False, compare=False)
    _deadline_monotonic: float = field(default=float('inf'), init=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.server_epoch = str(self.server_epoch or '').strip()
        self.env_id = str(self.env_id or '').strip()
        self.lease_id = str(self.lease_id or '').strip()
        self.client_id = str(self.client_id or '').strip()
        self.acquire_request_id = str(self.acquire_request_id or '').strip()
        self.release_request_id = str(self.release_request_id or '').strip()
        self.lease_generation = int(self.lease_generation)
        self.lease_ttl_s = float(self.lease_ttl_s)
        if self.heartbeat_interval_s is not None:
            self.heartbeat_interval_s = float(self.heartbeat_interval_s)

        missing = [
            name for name in (
                'server_epoch',
                'env_id',
                'lease_id',
                'client_id',
                'acquire_request_id',
                'release_request_id',
            ) if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"LeaseHandle is missing required fields: {', '.join(missing)}")
        if self.lease_generation <= 0:
            raise ValueError('LeaseHandle.lease_generation must be positive')
        if self.lease_ttl_s < 0:
            raise ValueError('LeaseHandle.lease_ttl_s must not be negative')
        if self.heartbeat_interval_s is not None and self.heartbeat_interval_s <= 0:
            raise ValueError('LeaseHandle.heartbeat_interval_s must be positive when set')
        self._set_deadline(self.lease_ttl_s)

    @classmethod
    def from_acquire_response(
        cls,
        payload: Mapping[str, Any],
        *,
        client_id: str,
        acquire_request_id: str,
        release_request_id: str,
    ) -> 'LeaseHandle':
        """Validate the flat v2 acquire identity returned by the server."""

        try:
            ttl = float(payload['lease_ttl_s'])
            generation = int(payload['lease_generation'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('AgentArk v2 acquire response has an invalid lease identity') from exc
        interval_value = payload.get('heartbeat_interval_s')
        interval = None if interval_value is None else float(interval_value)
        return cls(
            server_epoch=str(payload.get('server_epoch') or ''),
            env_id=str(payload.get('env_id') or ''),
            lease_id=str(payload.get('lease_id') or ''),
            lease_generation=generation,
            lease_ttl_s=ttl,
            heartbeat_interval_s=interval,
            client_id=client_id,
            acquire_request_id=acquire_request_id,
            release_request_id=release_request_id,
        )

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.server_epoch, self.env_id, self.lease_generation, self.lease_id)

    @property
    def heartbeat_enabled(self) -> bool:
        return self.lease_ttl_s > 0 and self.heartbeat_interval_s is not None

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._expired

    @property
    def expired_reason(self) -> str | None:
        with self._lock:
            return self._expired_reason

    @property
    def deadline_monotonic(self) -> float:
        with self._lock:
            return self._deadline_monotonic

    def identity(self) -> dict[str, Any]:
        return {
            'server_epoch': self.server_epoch,
            'env_id': self.env_id,
            'lease_id': self.lease_id,
            'lease_generation': self.lease_generation,
        }

    def request_identity(self) -> dict[str, Any]:
        """Identity fields for a path that already carries ``env_id``."""

        identity = self.identity()
        identity.pop('env_id')
        return identity

    def heartbeat_payload(self, heartbeat_id: str) -> dict[str, Any]:
        return {**self.identity(), 'heartbeat_id': heartbeat_id}

    def mark_alive(
        self,
        *,
        lease_ttl_s: float | None = None,
        lease_expires_in_s: float | None = None,
    ) -> None:
        """Refresh the local safety deadline after a successful server call."""

        with self._lock:
            if self._expired:
                return
            if lease_ttl_s is not None:
                ttl = float(lease_ttl_s)
                if ttl >= 0:
                    self.lease_ttl_s = ttl
            expires_in = self.lease_ttl_s if lease_expires_in_s is None else float(lease_expires_in_s)
            self._deadline_monotonic = (
                float('inf') if self.lease_ttl_s == 0 else time.monotonic() + max(0.0, expires_in))

    def mark_expired(self, reason: str) -> None:
        with self._lock:
            if self._expired:
                return
            self._expired = True
            self._expired_reason = str(reason or 'lease expired')

    def expire_if_deadline_elapsed(self, now: float | None = None) -> bool:
        resolved_now = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._expired:
                return True
            if resolved_now < self._deadline_monotonic:
                return False
            self._expired = True
            self._expired_reason = 'lease heartbeat deadline elapsed locally'
            return True

    def assert_active(self) -> None:
        if self.owner_pid != os.getpid():
            raise LeaseExpiredError(f'AgentArk lease env_id={self.env_id!r} belongs to process {self.owner_pid}, '
                                    f'not forked process {os.getpid()}')
        self.expire_if_deadline_elapsed()
        with self._lock:
            if self._expired:
                raise LeaseExpiredError(f'AgentArk lease env_id={self.env_id!r} is no longer active: '
                                        f"{self._expired_reason or 'unknown reason'}")

    def _set_deadline(self, expires_in_s: float) -> None:
        self._deadline_monotonic = (
            float('inf') if self.lease_ttl_s == 0 else time.monotonic() + max(0.0, expires_in_s))


@dataclass
class _Registration:
    handle: LeaseHandle
    base_url: str
    timeout_s: float
    next_due: float


class HeartbeatSupervisor:
    """One synchronous daemon batching all protocol-v2 leases in a process."""

    def __init__(self, *, default_timeout_s: float = 5.0) -> None:
        if default_timeout_s <= 0:
            raise ValueError('default_timeout_s must be positive')
        self._pid = os.getpid()
        self._default_timeout_s = float(default_timeout_s)
        self._condition = threading.Condition()
        self._registrations: dict[tuple[str, str, str, int, str], _Registration] = {}
        self._thread: threading.Thread | None = None
        self._shutdown = False

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    def register(
        self,
        handle: LeaseHandle,
        base_url: str,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self._assert_current_process()
        if not handle.heartbeat_enabled:
            return
        handle.assert_active()
        resolved_url = str(base_url or '').strip().rstrip('/')
        if not resolved_url:
            raise ValueError('heartbeat base_url must not be empty')
        resolved_timeout = self._default_timeout_s if timeout_s is None else float(timeout_s)
        if resolved_timeout <= 0:
            raise ValueError('heartbeat timeout_s must be positive')
        interval = self._effective_interval(handle)
        registration = _Registration(
            handle=handle,
            base_url=resolved_url,
            timeout_s=resolved_timeout,
            next_due=time.monotonic() + interval,
        )
        key = self._registration_key(resolved_url, handle)
        with self._condition:
            if self._shutdown:
                raise RuntimeError('HeartbeatSupervisor is shut down')
            self._registrations[key] = registration
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name='agentark-lease-heartbeats',
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()

    def unregister(self, handle: LeaseHandle, base_url: str | None = None) -> None:
        # A daemon thread inherited through fork is gone in the child.  Do not
        # mutate the parent's conceptual registration from that child.
        if self._pid != os.getpid():
            return
        with self._condition:
            if base_url is None:
                doomed = [key for key, item in self._registrations.items() if item.handle is handle]
                for key in doomed:
                    self._registrations.pop(key, None)
            else:
                self._registrations.pop(
                    self._registration_key(str(base_url).rstrip('/'), handle),
                    None,
                )
            self._condition.notify_all()

    def shutdown(self, *, join_timeout_s: float = 5.0) -> None:
        if self._pid != os.getpid():
            return
        with self._condition:
            self._shutdown = True
            self._registrations.clear()
            thread = self._thread
            self._condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(join_timeout_s)))

    def _assert_current_process(self) -> None:
        if self._pid != os.getpid():
            raise RuntimeError('HeartbeatSupervisor cannot be reused after fork')

    @staticmethod
    def _registration_key(
        base_url: str,
        handle: LeaseHandle,
    ) -> tuple[str, str, str, int, str]:
        return (base_url, handle.server_epoch, handle.env_id, handle.lease_generation, handle.lease_id)

    @staticmethod
    def _effective_interval(handle: LeaseHandle) -> float:
        assert handle.heartbeat_interval_s is not None
        # Never schedule later than one third of the current TTL.  A tiny
        # positive floor prevents a malformed server value from busy-spinning.
        ttl_interval = handle.lease_ttl_s / 3.0
        return max(0.01, min(float(handle.heartbeat_interval_s), ttl_interval))

    def _run(self) -> None:
        clients: dict[str, requests.Session] = {}
        try:
            while True:
                due = self._wait_for_due()
                if due is None:
                    return
                grouped: dict[str, list[_Registration]] = {}
                for registration in due:
                    grouped.setdefault(registration.base_url, []).append(registration)
                for base_url, registrations in grouped.items():
                    client = clients.get(base_url)
                    if client is None:
                        client = requests.Session()
                        clients[base_url] = client
                    self._heartbeat_group(client, base_url, registrations)
        finally:
            for client in clients.values():
                try:
                    client.close()
                except Exception:
                    logger.debug('Failed to close heartbeat HTTP client', exc_info=True)

    def _wait_for_due(self) -> list[_Registration] | None:
        with self._condition:
            while True:
                if self._shutdown:
                    return None
                now = time.monotonic()
                expired_keys = [
                    key for key, registration in self._registrations.items()
                    if registration.handle.expire_if_deadline_elapsed(now)
                ]
                for key in expired_keys:
                    self._registrations.pop(key, None)
                if not self._registrations:
                    self._condition.wait()
                    continue
                due = [
                    registration for registration in self._registrations.values()
                    # Coalesce leases whose deadlines are only a few
                    # milliseconds apart into one request per server URL.
                    if registration.next_due <= now + 0.01
                ]
                if due:
                    for registration in due:
                        registration.next_due = now + self._effective_interval(registration.handle)
                    return due
                next_wakeup = min(
                    min(registration.next_due, registration.handle.deadline_monotonic)
                    for registration in self._registrations.values())
                self._condition.wait(timeout=max(0.001, next_wakeup - now))

    def _heartbeat_group(
        self,
        client: requests.Session,
        base_url: str,
        registrations: list[_Registration],
    ) -> None:
        heartbeat_ids = [f'heartbeat-{uuid.uuid4().hex}' for _ in registrations]
        leases = [
            registration.handle.heartbeat_payload(heartbeat_id)
            for registration, heartbeat_id in zip(registrations, heartbeat_ids)  # noqa: B905
        ]
        timeout_s = max(registration.timeout_s for registration in registrations)
        try:
            response = client.post(
                f'{base_url}/v2/leases/heartbeat',
                json={'leases': leases},
                timeout=timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError('heartbeat response must be an object')
            items = payload.get('items')
            if not isinstance(items, list):
                raise ValueError('heartbeat response items must be a list')
        except Exception as exc:
            # A lost response is ambiguous.  The next scheduled batch uses a
            # fresh heartbeat id; the local TTL deadline is the final fence.
            logger.warning(
                'AgentArk lease heartbeat failed for %s (%d leases): %s: %s',
                registrations[0].base_url,
                len(registrations),
                type(exc).__name__,
                exc,
            )
            return

        response_epoch = str(payload.get('server_epoch') or '')
        if not response_epoch:
            logger.warning(
                'AgentArk lease heartbeat response from %s omitted server_epoch',
                registrations[0].base_url,
            )
            return
        for index, (registration, heartbeat_id) in enumerate(zip(registrations, heartbeat_ids)):  # noqa: B905
            handle = registration.handle
            if response_epoch and response_epoch != handle.server_epoch:
                handle.mark_expired(f'server epoch changed from {handle.server_epoch!r} to {response_epoch!r}')
                self.unregister(handle, registration.base_url)
                continue
            item = self._find_item(items, index, handle, heartbeat_id)
            if item is None:
                continue
            if bool(item.get('ok')):
                if not self._success_identity_matches(item, handle, heartbeat_id):
                    handle.mark_expired('heartbeat response returned a mismatched lease identity')
                    self.unregister(handle, registration.base_url)
                    continue
                handle.mark_alive(
                    lease_ttl_s=self._optional_number(item.get('lease_ttl_s')),
                    lease_expires_in_s=self._optional_number(item.get('lease_expires_in_s')),
                )
                continue
            error = item.get('error')
            error = error if isinstance(error, Mapping) else {}
            code = str(error.get('code') or 'heartbeat_failed')
            retryable = bool(error.get('retryable', False))
            if not retryable:
                handle.mark_expired(f"heartbeat rejected with {code}: {error.get('message', '')}")
                self.unregister(handle, registration.base_url)

    @staticmethod
    def _find_item(
        items: list[Any],
        index: int,
        handle: LeaseHandle,
        heartbeat_id: str,
    ) -> Mapping[str, Any] | None:
        for raw_item in items:
            if isinstance(raw_item, Mapping) and raw_item.get('heartbeat_id') == heartbeat_id:
                return raw_item
        # Failed items intentionally contain only env_id and error.  The server
        # preserves request order, so use that before the less-specific env id.
        if index < len(items) and isinstance(items[index], Mapping):
            return items[index]
        for raw_item in items:
            if isinstance(raw_item, Mapping) and raw_item.get('env_id') == handle.env_id:
                return raw_item
        return None

    @staticmethod
    def _success_identity_matches(
        item: Mapping[str, Any],
        handle: LeaseHandle,
        heartbeat_id: str,
    ) -> bool:
        try:
            generation = int(item.get('lease_generation'))
        except (TypeError, ValueError):
            return False
        return (str(item.get('server_epoch') or '') == handle.server_epoch
                and str(item.get('env_id') or '') == handle.env_id
                and str(item.get('lease_id') or '') == handle.lease_id and generation == handle.lease_generation
                and str(item.get('heartbeat_id') or '') == heartbeat_id)

    @staticmethod
    def _optional_number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


_SUPERVISOR_LOCK = threading.Lock()
_SUPERVISOR: HeartbeatSupervisor | None = None
_SUPERVISOR_PID: int | None = None


def get_heartbeat_supervisor() -> HeartbeatSupervisor:
    """Return the one lazy, fork-safe supervisor for the current process."""

    global _SUPERVISOR, _SUPERVISOR_PID
    pid = os.getpid()
    with _SUPERVISOR_LOCK:
        if _SUPERVISOR is None or _SUPERVISOR_PID != pid:
            _SUPERVISOR = HeartbeatSupervisor()
            _SUPERVISOR_PID = pid
        return _SUPERVISOR
