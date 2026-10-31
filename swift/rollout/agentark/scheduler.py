# Copyright (c) ModelScope Contributors. All rights reserved.
"""AgentArk behavior mixed into Swift's generic gym scheduler."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Mapping

from swift.infer_engine.protocol import ChatCompletionResponseChoice, RolloutInferRequest
from .client import AgentArkStaleLeaseError
from .env import AgentArkEnv
from .messages import new_environment_messages


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class AgentArkSchedulerMixin:
    """Drive AgentArk while preserving rollout-produced IDs and inline media."""

    def __init__(self, infer_engine=None, max_turns: int | None = None, **kwargs: Any) -> None:
        super().__init__(infer_engine, max_turns, **kwargs)
        self._step_infos: dict[str, list[dict[str, Any]]] = {}
        self._reset_infos: dict[str, dict[str, Any]] = {}
        self._pending_messages: dict[str, list[dict[str, Any]]] = {}
        self._loss_scopes: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def _create_env(self, env_config: dict[str, Any]) -> AgentArkEnv:
        return AgentArkEnv(env_config)

    async def on_trajectory_start(self, requests: list[RolloutInferRequest]) -> None:
        """Acquire all leases and replace dataset tickets with AgentArk messages."""

        async def _init_single(req: RolloutInferRequest) -> BaseException | None:
            uuid = req.uuid
            if not uuid:
                return ValueError('RolloutInferRequest.uuid is required by AgentArkScheduler')
            if uuid in self._envs:
                await self.finalize_trajectory(uuid, reason='reinitialized')

            env_config = (req.data_dict or {}).get('env_config', {})
            if not isinstance(env_config, dict):
                return ValueError('RolloutInferRequest.data_dict.env_config must be an object')
            try:
                env = self._create_env(env_config)
                if not isinstance(env, AgentArkEnv):
                    # Tests and advanced callers may provide a compatible subclass,
                    # but an unrelated Swift Env cannot expose AgentArk messages.
                    required = ('initial_messages', 'pending_messages', 'assistant_loss_scope')
                    if not all(hasattr(env, name) for name in required):
                        raise TypeError('AgentArkScheduler requires an AgentArkEnv-compatible environment')
                self._envs[uuid] = env
                self._total_rewards[uuid] = 0.0
                self._step_rewards[uuid] = []
                self._step_infos[uuid] = []
                self._pending_obs[uuid] = None
                self._pending_messages[uuid] = []
                await env.reset(req)
                req.messages = deepcopy(env.initial_messages)
                # AgentArk uses only OpenAI inline media blocks. Clearing ticket
                # media prevents Swift TemplateInputs from seeing both forms.
                req.images = []
                req.audios = []
                req.videos = []
                self._reset_infos[uuid] = _json_safe(env.reset_info)
                self._loss_scopes[uuid] = env.assistant_loss_scope
                self._metadata[uuid] = {
                    'agentark_env_id':
                    env.env_id,
                    'agentark_unity_id':
                    env.unity_id,
                    'group_uid':
                    env.group_uid,
                    'agentark_server_epoch':
                    (env.lease.server_epoch if getattr(env, 'lease', None) is not None else None),
                    'agentark_lease_generation':
                    (env.lease.lease_generation if getattr(env, 'lease', None) is not None else None),
                }
            except BaseException as exc:  # noqa: B036
                await self.finalize_trajectory(uuid, reason='reset_error')
                return exc
            return None

        results = await asyncio.gather(*[_init_single(req) for req in requests])
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            # A rollout batch is not usable when one ticket failed to reset. Do
            # not leave the successfully initialized sibling leases behind.
            await asyncio.gather(
                *[self.finalize_trajectory(req.uuid, reason='batch_reset_error') for req in requests if req.uuid])
            raise errors[0]

    def _rollout_infos(
        self,
        uuid: str,
        *,
        gym_done: bool,
        termination_reason: str | None = None,
        finish_reason: str | None = None,
        end_turn: int | None = None,
    ) -> dict[str, Any]:
        infos: dict[str, Any] = {
            **deepcopy(self._metadata.get(uuid, {})),
            'reset_info': deepcopy(self._reset_infos.get(uuid, {})),
            'total_reward': float(self._total_rewards.get(uuid, 0.0)),
            'step_rewards': list(self._step_rewards.get(uuid, [])),
            'step_infos': deepcopy(self._step_infos.get(uuid, [])),
            'gym_done': bool(gym_done),
            'termination_reason': termination_reason or 'unknown',
            'finish_reason': finish_reason or 'unknown',
            'end_turn': end_turn if end_turn is not None else len(self._step_rewards.get(uuid, [])),
        }
        return _json_safe(infos)

    async def on_turn_end(
        self,
        infer_request: RolloutInferRequest,
        response_choice: ChatCompletionResponseChoice,
        current_turn: int,
    ) -> dict[str, Any]:
        uuid = infer_request.uuid
        if not uuid or uuid not in self._envs:
            return {
                'done': True,
                'rollout_infos': {
                    'gym_done': False,
                    'termination_reason': 'missing_env',
                    'finish_reason': response_choice.finish_reason or 'unknown',
                    'end_turn': current_turn,
                },
            }

        token_ids = response_choice.token_ids
        if token_ids is None:
            infos = await self.finalize_trajectory(uuid, reason='missing_response_token_ids')
            raise RuntimeError('AgentArkScheduler requires exact response_choice.token_ids; '
                               f'trajectory {uuid} was released. infos={infos}')

        # A length-truncated code/tool call is not a valid action. Releasing it
        # here also avoids GYMScheduler's done override bypassing check_finished.
        if response_choice.finish_reason == 'length':
            infos = self._rollout_infos(
                uuid,
                gym_done=False,
                termination_reason='length',
                finish_reason=response_choice.finish_reason,
                end_turn=current_turn,
            )
            finalized = await self.finalize_trajectory(uuid, reason='length')
            if 'release_error' in finalized:
                infos['release_error'] = finalized['release_error']
            return {'done': True, 'rollout_infos': infos}

        env = self._envs[uuid]
        try:
            _, reward, env_done, step_info = await env.step(
                deepcopy(infer_request.messages),
                action_id=f'{uuid}:{current_turn}',
                turn_index=current_turn,
            )
        except AgentArkStaleLeaseError as exc:
            infos = await self.finalize_trajectory(uuid, reason='stale_lease')
            infos['trajectory_invalid'] = True
            infos['lease_recovery'] = 'discard_trajectory'
            infos['stale_lease_error'] = str(exc)
            infos['finish_reason'] = response_choice.finish_reason or 'unknown'
            infos['end_turn'] = current_turn
            return {'done': True, 'rollout_infos': _json_safe(infos)}
        except BaseException:
            await self.finalize_trajectory(uuid, reason='step_error')
            raise

        reward = float(reward)
        safe_step_info = _json_safe(step_info)
        self._total_rewards[uuid] = self._total_rewards.get(uuid, 0.0) + reward
        self._step_rewards.setdefault(uuid, []).append(reward)
        self._step_infos.setdefault(uuid, []).append(safe_step_info)

        hit_max_turns = bool(self.max_turns and current_turn >= self.max_turns)
        should_stop = bool(env_done or hit_max_turns)
        self._pending_messages[uuid] = [] if should_stop else new_environment_messages(
            infer_request.messages,
            env.pending_messages,
        )

        termination_reason = None
        if env_done:
            termination_reason = 'env_done'
        elif hit_max_turns:
            termination_reason = 'max_turns'
        infos = self._rollout_infos(
            uuid,
            gym_done=bool(env_done),
            termination_reason=termination_reason,
            finish_reason=response_choice.finish_reason,
            end_turn=current_turn,
        )
        if should_stop:
            finalized = await self.finalize_trajectory(uuid, reason=termination_reason or 'finished')
            if 'release_error' in finalized:
                infos['release_error'] = finalized['release_error']
        return {'done': should_stop, 'rollout_infos': infos}

    def step(
        self,
        infer_request: RolloutInferRequest,
        response_choice: ChatCompletionResponseChoice,
        current_turn: int,
    ) -> dict[str, Any]:
        """Append only environment messages and return this turn's exact IDs."""

        uuid = infer_request.uuid
        if not uuid:
            raise ValueError('RolloutInferRequest.uuid is required by AgentArkScheduler')
        for message in self._pending_messages.pop(uuid, []):
            infer_request.messages.append(deepcopy(message))

        token_ids = response_choice.token_ids
        if token_ids is None:
            # on_turn_end validates this first in the supported Swift drivers.
            raise RuntimeError('AgentArkScheduler requires exact response_choice.token_ids')
        ids = list(token_ids)
        loss_scope = self._loss_scopes.get(uuid, 'all_turns')
        mask_value = 0 if loss_scope == 'last_round' else 1
        return {
            'infer_request': infer_request,
            'response_token_ids': ids,
            'response_loss_mask': [mask_value] * len(ids),
        }

    async def finalize_trajectory(self, uuid: str, *, reason: str) -> dict[str, Any]:
        """Public best-effort cleanup hook for errors, cancellation, and tests."""

        infos = self._rollout_infos(uuid, gym_done=False, termination_reason=reason)
        env = self._envs.pop(uuid, None)
        if env is not None:
            await env.close()
            if getattr(env, 'close_error', None):
                infos['release_error'] = env.close_error
        self._total_rewards.pop(uuid, None)
        self._step_rewards.pop(uuid, None)
        self._step_infos.pop(uuid, None)
        self._reset_infos.pop(uuid, None)
        self._pending_obs.pop(uuid, None)
        self._pending_messages.pop(uuid, None)
        self._loss_scopes.pop(uuid, None)
        self._metadata.pop(uuid, None)
        return _json_safe(infos)

    async def finalize_all(self, *, reason: str = 'scheduler_shutdown') -> None:
        """Release every live trajectory known to this scheduler."""

        await asyncio.gather(*[self.finalize_trajectory(uuid, reason=reason) for uuid in list(self._envs)])

    async def on_trajectory_end(
        self,
        requests: list[RolloutInferRequest],
        error: BaseException | None = None,
    ) -> None:
        """Release leases left alive at the rollout lifecycle boundary."""

        reason = 'rollout_error' if error is not None else 'rollout_boundary'
        await asyncio.gather(
            *[self.finalize_trajectory(request.uuid, reason=reason) for request in requests if request.uuid])

    async def _close_and_remove(self, uuid: str) -> None:
        # Preserve GYMScheduler's helper name for callers in ms-swift.
        await self.finalize_trajectory(uuid, reason='closed')


__all__ = ['AgentArkSchedulerMixin']
