from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

IMAGE_URL = 'data:image/png;base64,iVBORw0KGgo='


def initial_messages(label: str = 'initial') -> list[dict[str, Any]]:
    return [
        {
            'role': 'system',
            'content': 'AgentArk system prompt'
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': label
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': IMAGE_URL
                    }
                },
            ],
        },
    ]


def delta_messages(assistant: str, label: str = 'next') -> list[dict[str, Any]]:
    return [
        {
            'role': 'assistant',
            'content': assistant
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': label
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': IMAGE_URL
                    }
                },
            ],
        },
    ]


def make_request(*, uuid: str, group_uid: str, loss_scope: str = 'all_turns'):
    from swift.infer_engine.protocol import RolloutInferRequest

    env_config = {
        'name': 'agentark',
        'group_uid': group_uid,
        'assistant_loss_scope': loss_scope,
        'server_url': 'http://unused.test',
        'agentark_env_cfg': {
            'runtime': 'fake'
        },
    }
    return RolloutInferRequest(
        messages=[{
            'role': 'user',
            'content': f'<agentark-ticket:{group_uid}>'
        }],
        data_dict={'env_config': env_config},
        uuid=uuid,
    )


def make_choice(
    content: str,
    *,
    token_ids: list[int] | None = None,
    finish_reason: str = 'stop',
):
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        token_ids=list(token_ids or [101, 102]),
        finish_reason=finish_reason,
        logprobs=None,
    )


class FakeAgentArkClient:
    """In-memory transport with the same async surface as AgentArkHttpClient."""

    def __init__(
        self,
        *,
        step_payloads: list[Any] | None = None,
        protocol_version: str = 'v2',
    ):
        self.protocol_version = protocol_version
        self.client_id = 'fake-swift-client'
        self.base_url = 'http://unused.test'
        self.acquire_calls: list[dict[str, Any]] = []
        self.step_calls: list[dict[str, Any]] = []
        self.release_calls: list[str] = []
        self.release_payloads: list[dict[str, Any]] = []
        self.aclose_calls = 0
        self._step_payloads = list(step_payloads or [])
        self._next_env = 1
        self.heartbeat_supervisor = FakeHeartbeatSupervisor()

    async def acquire_start(
        self,
        env_cfg: dict[str, Any],
        *,
        uid: str,
        task_name: str | None = None,
        group_seed: int | None = None,
        env_id: str | None = None,
        unity_env_id: int | None = None,
        acquire_request_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        call = {
            'env_cfg': copy.deepcopy(env_cfg),
            'uid': uid,
            'task_name': task_name,
            'group_seed': group_seed,
            'env_id': env_id,
            'unity_env_id': unity_env_id,
            'acquire_request_id': acquire_request_id,
            'client_id': client_id,
        }
        self.acquire_calls.append(call)
        leased_env_id = env_id or f'env-{self._next_env}'
        self._next_env += 1
        response = {
            'env_id': leased_env_id,
            'unity_id': 0,
            'obs': {
                'messages': initial_messages(f'initial:{uid}')
            },
            'info': {
                'task_name': f'task-for:{uid}',
                'rollout_group_seed': 31415,
            },
        }
        if self.protocol_version == 'v2':
            response.update({
                'server_epoch': 'fake-server-epoch',
                'lease_id': f'lease-capability-{leased_env_id}',
                'lease_generation': 1,
                'lease_ttl_s': 60.0,
                'heartbeat_interval_s': 20.0,
                'acquire_request_id': acquire_request_id,
                'replayed': False,
            })
        return response

    async def step(
        self,
        lease,
        *,
        action: str,
        assistant: str,
        action_id: str | None = None,
        turn_index: int | None = None,
    ) -> dict[str, Any]:
        env_id = lease.env_id if hasattr(lease, 'env_id') else str(lease)
        call = {
            'env_id': env_id,
            'action': action,
            'assistant': assistant,
            'action_id': action_id,
            'turn_index': turn_index,
        }
        if hasattr(lease, 'identity'):
            call.update(lease.identity())
        self.step_calls.append(call)
        if not self._step_payloads:
            return {
                'unity_id': 0,
                'obs': {
                    'messages': delta_messages(assistant)
                },
                'reward': 0.25,
                'done': False,
                'info': {
                    'status': 'running'
                },
            }
        result = self._step_payloads.pop(0)
        if isinstance(result, BaseException):
            raise result
        return copy.deepcopy(result)

    async def release(self, lease, *, release_request_id: str | None = None) -> dict[str, Any]:
        env_id = lease.env_id if hasattr(lease, 'env_id') else str(lease)
        self.release_calls.append(env_id)
        payload = {'env_id': env_id, 'release_request_id': release_request_id}
        if hasattr(lease, 'identity'):
            payload.update(lease.identity())
        self.release_payloads.append(payload)
        return {'ok': True}

    async def aclose(self) -> None:
        self.aclose_calls += 1


class FakeHeartbeatSupervisor:

    def __init__(self):
        self.register_calls: list[dict[str, Any]] = []
        self.unregister_calls: list[dict[str, Any]] = []

    def register(self, lease, base_url: str, *, timeout_s: float | None = None) -> None:
        self.register_calls.append({'lease': lease, 'base_url': base_url, 'timeout_s': timeout_s})

    def unregister(self, lease, base_url: str | None = None) -> None:
        self.unregister_calls.append({'lease': lease, 'base_url': base_url})


def append_generated_assistant(request, content: str) -> None:
    request.messages.append({'role': 'assistant', 'content': content})
