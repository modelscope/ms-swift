# Copyright (c) ModelScope Contributors. All rights reserved.
"""Swift Env implementation backed by the existing AgentArk HTTP server."""

from __future__ import annotations

import logging
import os
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from swift.infer_engine.protocol import RolloutInferRequest
from swift.rollout.gym_env import Env
from swift.template import Messages
from .client import AgentArkHttpClient, AgentArkStaleLeaseError, stable_operation_id
from .heartbeat import HeartbeatSupervisor, LeaseExpiredError, LeaseHandle, get_heartbeat_supervisor
from .messages import copy_messages, extract_action, latest_assistant_text

logger = logging.getLogger(__name__)


def _optional_float(value: Any, *, default: float, name: str) -> float:
    resolved = default if value in (None, '') else float(value)
    if resolved <= 0:
        raise ValueError(f'{name} must be positive')
    return resolved


def _expand_paths_and_vars(value: Any) -> Any:
    """Recursively expand shell variables and ``~`` in config strings."""

    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, Mapping):
        return {str(key): _expand_paths_and_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_paths_and_vars(item) for item in value]
    if isinstance(value, tuple):
        return [_expand_paths_and_vars(item) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; non-mapping override values replace base."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_runtime_config(path_value: str | os.PathLike[str] | None) -> dict[str, Any]:
    if path_value in (None, ''):
        return {}
    path = Path(str(path_value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'AgentArk runtime config not found: {path}')
    with path.open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f'AgentArk runtime config must contain a mapping: {path}')
    expanded = _expand_paths_and_vars(data)
    assert isinstance(expanded, dict)
    return expanded


def resolve_runtime_config(env_config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the file's ``env_cfg`` plus an optional per-row deep override."""

    config_path = env_config.get('runtime_config_path') or os.getenv('AGENTARK_RUNTIME_CONFIG')
    document = _load_runtime_config(config_path)
    # The shared AgentArk runtime file also contains server/warmup/interaction
    # sections. acquire_start expects only its nested env_cfg payload so it can
    # match a pre-warmed pool runtime.
    file_env_cfg = document.get('env_cfg', document)
    if not isinstance(file_env_cfg, Mapping):
        raise ValueError('AgentArk runtime config env_cfg must be a mapping')
    result = deepcopy(dict(file_env_cfg))
    inline = env_config.get('agentark_env_cfg')
    if inline is None:
        inline = env_config.get('runtime_cfg')
    if inline is None:
        inline = env_config.get('cfg')
    if inline is not None:
        if not isinstance(inline, Mapping):
            raise ValueError('env_config.agentark_env_cfg must be a mapping')
        result = _deep_merge(result, _expand_paths_and_vars(inline))
    if not result:
        raise ValueError('AgentArk runtime config is empty; set AGENTARK_RUNTIME_CONFIG or '
                         'env_config.agentark_env_cfg')
    return result


class AgentArkEnv(Env):
    """One Swift rollout trajectory backed by one leased AgentArk runtime."""

    def __init__(
        self,
        env_config: dict[str, Any],
        *,
        client: AgentArkHttpClient | Any | None = None,
        heartbeat_supervisor: HeartbeatSupervisor | Any | None = None,
    ) -> None:
        super().__init__(deepcopy(env_config))
        self.agentark_env_cfg = resolve_runtime_config(env_config)
        self.group_uid = str(env_config.get('group_uid') or env_config.get('uid') or '').strip()
        if not self.group_uid:
            raise ValueError('env_config.group_uid is required for GRPO group-consistent reset')

        loss_scope = env_config.get('assistant_loss_scope') or os.getenv('AGENTARK_ASSISTANT_LOSS_SCOPE', 'all_turns')
        self.assistant_loss_scope = str(loss_scope).strip().lower()
        if self.assistant_loss_scope not in {'all_turns', 'last_round'}:
            raise ValueError("assistant_loss_scope must be 'all_turns' or 'last_round'")

        self.task_name = env_config.get('task_name')
        seed = env_config.get('group_seed')
        self.group_seed = None if seed in (None, '') else int(seed)
        self.requested_env_id = env_config.get('env_id')
        unity_env_id = env_config.get('unity_env_id')
        self.unity_env_id = None if unity_env_id in (None, '') else int(unity_env_id)

        server_url = env_config.get('server_url') or os.getenv('AGENTARK_SERVER_URL', 'http://127.0.0.1:18080')
        timeout_s = _optional_float(
            env_config.get('http_timeout_s') or os.getenv('AGENTARK_HTTP_TIMEOUT'),
            default=600.0,
            name='http_timeout_s',
        )
        release_timeout_s = _optional_float(
            env_config.get('release_timeout_s') or os.getenv('AGENTARK_RELEASE_TIMEOUT'),
            default=30.0,
            name='release_timeout_s',
        )
        heartbeat_timeout_s = _optional_float(
            env_config.get('heartbeat_timeout_s') or os.getenv('AGENTARK_HEARTBEAT_TIMEOUT'),
            default=5.0,
            name='heartbeat_timeout_s',
        )
        configured_protocol = str(env_config.get('protocol_version') or os.getenv('AGENTARK_PROTOCOL_VERSION')
                                  or 'v2').strip().lower()
        if configured_protocol not in {'v1', 'v2'}:
            raise ValueError("protocol_version must be 'v1' or 'v2'")
        self._owns_client = client is None
        self.client = client or AgentArkHttpClient(
            str(server_url),
            timeout_s=timeout_s,
            release_timeout_s=release_timeout_s,
            protocol_version=configured_protocol,
        )
        self.protocol_version = str(getattr(self.client, 'protocol_version', configured_protocol)).lower()
        if self.protocol_version not in {'v1', 'v2'}:
            raise ValueError("client.protocol_version must be 'v1' or 'v2'")
        self.server_url = str(getattr(self.client, 'base_url', server_url)).rstrip('/')
        self.heartbeat_timeout_s = heartbeat_timeout_s
        # A fake/injected transport may provide its own deterministic
        # supervisor.  Production resolves the process singleton lazily after
        # acquire, which is safe when Env objects were created before a fork.
        self._heartbeat_supervisor = heartbeat_supervisor or getattr(self.client, 'heartbeat_supervisor', None)

        self.env_id: str | None = None
        self.unity_id: int | None = None
        self.lease: LeaseHandle | None = None
        self.acquire_request_id: str | None = None
        self.release_request_id: str | None = None
        self.initial_messages: Messages = []
        self.pending_messages: Messages = []
        self.reset_info: dict[str, Any] = {}
        self.close_error: str | None = None
        self._reset_called = False
        self._heartbeat_registered = False
        self._release_attempted = False
        self._closed = False

    async def reset(self, config: RolloutInferRequest) -> tuple[str, dict[str, Any], str]:
        if self._reset_called:
            raise RuntimeError('AgentArkEnv.reset may only be called once per trajectory')
        self._reset_called = True
        try:
            acquire_kwargs = {
                'uid': self.group_uid,
                'task_name': self.task_name,
                'group_seed': self.group_seed,
                'env_id': self.requested_env_id,
                'unity_env_id': self.unity_env_id,
            }
            if self.protocol_version == 'v2':
                trajectory_uuid = str(config.uuid or '').strip()
                if not trajectory_uuid:
                    raise ValueError('RolloutInferRequest.uuid is required for AgentArk protocol v2')
                client_id = str(getattr(self.client, 'client_id', '') or '').strip()
                if not client_id:
                    raise ValueError('AgentArk protocol v2 client_id must not be empty')
                # group_uid selects a shared GRPO task; request.uuid selects one
                # concrete trajectory lease and must never be replaced by it.
                self.acquire_request_id = stable_operation_id('acquire', client_id, trajectory_uuid)
                acquire_kwargs.update(
                    acquire_request_id=self.acquire_request_id,
                    client_id=client_id,
                )
            payload = await self.client.acquire_start(
                self.agentark_env_cfg,
                **acquire_kwargs,
            )
            env_id = payload.get('env_id')
            if not isinstance(env_id, str) or not env_id:
                raise ValueError('AgentArk acquire_start response is missing env_id')
            self.env_id = env_id
            if self.protocol_version == 'v2':
                assert self.acquire_request_id is not None
                echoed_acquire_id = payload.get('acquire_request_id')
                if (echoed_acquire_id is not None and str(echoed_acquire_id) != self.acquire_request_id):
                    raise ValueError('AgentArk v2 acquire response echoed a different acquire_request_id')
                client_id = str(getattr(self.client, 'client_id', '') or '')
                self.release_request_id = stable_operation_id(
                    'release',
                    self.acquire_request_id,
                    str(payload.get('server_epoch') or ''),
                    env_id,
                    str(payload.get('lease_generation') or ''),
                    str(payload.get('lease_id') or ''),
                )
                self.lease = LeaseHandle.from_acquire_response(
                    payload,
                    client_id=client_id,
                    acquire_request_id=self.acquire_request_id,
                    release_request_id=self.release_request_id,
                )
            unity_id = payload.get('unity_id')
            self.unity_id = int(unity_id) if unity_id is not None else None
            obs = payload.get('obs')
            if not isinstance(obs, Mapping):
                raise ValueError('AgentArk acquire_start response is missing obs')
            self.initial_messages = copy_messages(
                obs.get('messages'),
                field_name='acquire_start.obs.messages',
                require_nonempty=True,
            )
            info = payload.get('info') or {}
            if not isinstance(info, Mapping):
                raise ValueError('AgentArk acquire_start response info must be an object')
            self.reset_info = deepcopy(dict(info))
            if self.lease is not None and self.lease.heartbeat_enabled:
                if self._heartbeat_supervisor is None:
                    self._heartbeat_supervisor = get_heartbeat_supervisor()
                self._heartbeat_supervisor.register(
                    self.lease,
                    self.server_url,
                    timeout_s=self.heartbeat_timeout_s,
                )
                self._heartbeat_registered = True
        except BaseException:
            # If acquire succeeded but response validation failed, return the
            # lease before propagating the malformed-response error.
            await self.close()
            raise

        # The generic Env API only admits string observation/system fields.
        # AgentArkScheduler deliberately ignores these placeholders and copies
        # initial_messages directly, preserving inline image_url blocks.
        return '', deepcopy(self.reset_info), ''

    async def step(
        self,
        action: Messages,
        *,
        action_id: str | None = None,
        turn_index: int | None = None,
    ) -> tuple[str, float, bool, dict[str, Any]]:
        if self._closed or self.env_id is None:
            raise RuntimeError('AgentArkEnv.step called without an active lease')
        assistant_raw = latest_assistant_text(action)
        action_extracted = extract_action(assistant_raw)
        if self.protocol_version == 'v2':
            if self.lease is None:
                raise RuntimeError('AgentArkEnv.step called without a protocol-v2 LeaseHandle')
            self.lease.assert_active()
            payload = await self.client.step(
                self.lease,
                action=action_extracted,
                assistant=assistant_raw,
                action_id=action_id,
                turn_index=turn_index,
            )
            if self.lease.expired:
                raise LeaseExpiredError(self.lease.expired_reason
                                        or 'AgentArk server expired the lease while the step was in progress')
        else:
            payload = await self.client.step(
                self.env_id,
                action=action_extracted,
                assistant=assistant_raw,
            )
        obs = payload.get('obs') or {}
        if not isinstance(obs, Mapping):
            raise ValueError('AgentArk step response obs must be an object')
        raw_messages = obs.get('messages', [])
        self.pending_messages = copy_messages(
            raw_messages,
            field_name='step.obs.messages',
            require_nonempty=False,
        )
        info = payload.get('info') or {}
        if not isinstance(info, Mapping):
            raise ValueError('AgentArk step response info must be an object')
        try:
            reward = float(payload.get('reward', 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"AgentArk step reward is not numeric: {payload.get('reward')!r}") from exc
        done = bool(payload.get('done', False))
        return '', reward, done, deepcopy(dict(info))

    async def close(self) -> None:
        """Best-effort, locally idempotent release of this trajectory's lease."""

        if self._closed:
            return
        self._closed = True
        if self._heartbeat_registered and self.lease is not None:
            self._heartbeat_registered = False
            try:
                assert self._heartbeat_supervisor is not None
                self._heartbeat_supervisor.unregister(self.lease, self.server_url)
            except BaseException as exc:  # noqa: B036
                logger.warning(
                    'Failed to unregister AgentArk heartbeat env_id=%s: %s',
                    self.env_id,
                    exc,
                )
        release_target: LeaseHandle | str | None
        release_target = self.lease if self.protocol_version == 'v2' else self.env_id
        if release_target is not None and not self._release_attempted:
            self._release_attempted = True
            try:
                if self.protocol_version == 'v2':
                    assert isinstance(release_target, LeaseHandle)
                    if release_target.owner_pid != os.getpid():
                        raise RuntimeError('refusing to release a lease inherited across fork')
                    await self.client.release(
                        release_target,
                        release_request_id=self.release_request_id,
                    )
                else:
                    await self.client.release(release_target)
            except AgentArkStaleLeaseError:
                logger.info(
                    'AgentArk lease env_id=%s was already stale during release',
                    self.env_id,
                )
            except BaseException as exc:  # noqa: B036
                self.close_error = f'{type(exc).__name__}: {exc}'
                logger.warning('Failed to release AgentArk env_id=%s: %s', self.env_id, self.close_error)
        if self._owns_client:
            try:
                await self.client.aclose()
            except BaseException as exc:  # noqa: B036
                if self.close_error is None:
                    self.close_error = f'{type(exc).__name__}: {exc}'
                logger.warning('Failed to close AgentArk HTTP client: %s', exc)
