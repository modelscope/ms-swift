"""Swift scheduler adapter for Twinkle's shared multi-turn rollout engine."""
from __future__ import annotations
import asyncio
import copy
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from . import SHIFTED_KEY, RolloutSample, _sampled_token_logprobs


@dataclass
class _SchedulerState:
    request: Any
    extra: Dict[str, Any]


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _messages_with_token_ids(messages: List[dict], response_token_ids: List[List[int]],
                             response_loss_mask: List[List[int]]) -> List[dict]:
    """Replace generated assistant turns, newest first, with exact sampled token IDs."""
    result = copy.deepcopy(messages)
    if len(response_token_ids) != len(response_loss_mask):
        raise RuntimeError('response_token_ids and response_loss_mask must have the same number of turns.')
    completion_index = 0
    for message in reversed(result):
        if message.get('role') != 'assistant':
            continue
        if completion_index >= len(response_token_ids):
            break
        ids = list(response_token_ids[-1 - completion_index])
        mask = list(response_loss_mask[-1 - completion_index])
        if len(ids) != len(mask):
            raise RuntimeError('response token IDs and loss mask are misaligned within a turn.')
        message['content'] = {'token_ids': ids, 'loss_scale': mask}
        completion_index += 1
    if completion_index != len(response_token_ids):
        raise RuntimeError(
            f'multi-turn messages contain {completion_index} generated assistant turns, but rollout recorded '
            f'{len(response_token_ids)} token sequences.')
    return result


class _SwiftSchedulerController:
    """Translate legacy swift scheduler hooks to Twinkle's turn-controller protocol."""

    def __init__(self, template: Any, scheduler: Any):
        self.template = template
        self.scheduler = scheduler

    @staticmethod
    def _request(messages: List[dict], extra: Dict[str, Any]):
        from swift.infer_engine.protocol import RolloutInferRequest
        request_fields = ('images', 'audios', 'videos', 'tools', 'objects', 'chat_template_kwargs')
        kwargs = {key: copy.deepcopy(extra[key]) for key in request_fields if extra.get(key) is not None}
        return RolloutInferRequest(
            messages=copy.deepcopy(messages), data_dict=copy.deepcopy(extra), uuid=uuid.uuid4().hex, **kwargs)

    def create_state(self, trajectory: dict) -> _SchedulerState:
        extra = copy.deepcopy(trajectory.get('_swift_extra') or {})
        return _SchedulerState(self._request(trajectory['messages'], extra), extra)

    @staticmethod
    def sampler_input(state: _SchedulerState) -> dict:
        trajectory = {'messages': copy.deepcopy(state.request.messages)}
        for key in ('images', 'videos', 'audios', 'tools'):
            value = getattr(state.request, key, None)
            if value:
                trajectory[key] = copy.deepcopy(value)
        return trajectory

    def start(self, states: List[_SchedulerState]) -> None:
        _run_async(self.scheduler.on_trajectory_start([state.request for state in states]))

    @staticmethod
    def _choice(sequence: Any, logprobs: List[float]) -> Any:
        return SimpleNamespace(
            index=0,
            message=SimpleNamespace(content=sequence.decoded or ''),
            finish_reason=sequence.stop_reason,
            token_ids=list(sequence.tokens or []),
            logprobs={'content': [{'logprob': value} for value in logprobs]},
        )

    @staticmethod
    def _append_message(messages: List[dict], completion: str) -> bool:
        if messages and messages[-1].get('role') == 'assistant':
            if messages[-1].get('content') is None:
                messages.pop()
            else:
                messages[-1]['content'] += completion
                return True
        messages.append({'role': 'assistant', 'content': completion})
        return False

    async def _turn_ends(self, states: List[_SchedulerState], choices: List[Any], current_turn: int):
        return list(await asyncio.gather(*[
            self.scheduler.on_turn_end(state.request, choice, current_turn)
            for state, choice in zip(states, choices)
        ]))

    def on_turn(self, states: List[_SchedulerState], sequences: List[Any], current_turn: int,
                forced_stops: List[bool]) -> List[Dict[str, Any]]:
        choices, continuations = [], []
        for state, sequence in zip(states, sequences):
            token_ids = list(sequence.tokens or [])
            logprobs = _sampled_token_logprobs(token_ids, sequence.logprobs)
            if len(logprobs) != len(token_ids):
                raise RuntimeError(f'multi-turn rollout logprobs misaligned: {len(logprobs)} logprobs for '
                                   f'{len(token_ids)} sampled tokens.')
            choices.append(self._choice(sequence, logprobs))
            continuations.append(self._append_message(state.request.messages, sequence.decoded or ''))

        turn_results = _run_async(self._turn_ends(states, choices, current_turn))
        updates = []
        for state, choice, continuation, turn_result, forced_stop in zip(
                states, choices, continuations, turn_results, forced_stops):
            rollout_infos = dict(turn_result.get('rollout_infos') or {})
            stop = (bool(turn_result['done']) if 'done' in turn_result else
                    self.scheduler.check_finished(state.request, choice, current_turn))
            stop = stop or forced_stop
            token_ids = list(choice.token_ids or [])
            loss_mask = [1] * len(token_ids)
            logprobs = [item['logprob'] for item in choice.logprobs.get('content', [])]
            if not stop:
                step_result = self.scheduler.step(state.request, choice, current_turn)
                state.request = step_result['infer_request']
                rollout_infos.update(step_result.get('rollout_infos') or {})
                if 'response_token_ids' in step_result:
                    token_ids = list(step_result['response_token_ids'])
                    loss_mask = list(step_result.get('response_loss_mask', [1] * len(token_ids)))
                if step_result.get('rollout_logprobs'):
                    logprobs = list(step_result['rollout_logprobs'])
            updates.append({
                'done': stop,
                'continuation': continuation,
                'response_token_ids': token_ids,
                'response_loss_mask': loss_mask,
                'rollout_logprobs': logprobs,
                'rollout_infos': rollout_infos,
                'stop_reason': choice.finish_reason,
            })
        return updates

    def finalize(self, state: _SchedulerState, response_token_ids: List[List[int]],
                 response_loss_mask: List[List[int]]) -> Dict[str, Any]:
        messages = _messages_with_token_ids(state.request.messages, response_token_ids, response_loss_mask)
        row = copy.deepcopy(state.extra)
        row['messages'] = messages
        row['add_eos'] = False
        template = copy.copy(self.template)
        if hasattr(template, 'set_mode'):
            template.set_mode('train')
        encoded = template.encode(row)
        if not isinstance(encoded, dict) or encoded.get('labels') is None:
            raise RuntimeError('multi-turn training template returned no labels.')
        encoded[SHIFTED_KEY] = True
        decoded = ''
        for message in reversed(state.request.messages):
            if message.get('role') == 'assistant':
                content = message.get('content')
                decoded = content if isinstance(content, str) else ''
                break
        return {'encoded': encoded, 'messages': copy.deepcopy(state.request.messages), 'decoded': decoded}

    def close(self, states: List[_SchedulerState]) -> None:
        closer = getattr(self.scheduler, '_close_and_remove', None)
        if closer is None:
            return

        async def _close_all():
            await asyncio.gather(*(closer(state.request.uuid) for state in states))

        _run_async(_close_all())


class MultiTurnRollout:
    """Thin Swift adapter around :class:`twinkle_agentic.rollout.MultiTurnRollout`."""

    def __init__(self,
                 sampler: Any,
                 template: Any,
                 scheduler: Any,
                 *,
                 max_turns: Optional[int] = None,
                 gym_env: Optional[str] = None,
                 completion_length_limit_scope: str = 'per_round'):
        from twinkle.data_format import SamplingParams
        from twinkle_agentic.rollout import MultiTurnRollout as TwinkleMultiTurnRollout

        self.scheduler = self._resolve_scheduler(scheduler, template, max_turns=max_turns, gym_env=gym_env)
        self.controller = _SwiftSchedulerController(template, self.scheduler)
        self.rollout = TwinkleMultiTurnRollout(
            sampler,
            template,
            sampling_params=SamplingParams(),
            max_turns=max_turns,
            turn_controller=self.controller,
            completion_length_limit_scope=completion_length_limit_scope)

    @staticmethod
    def _resolve_scheduler(scheduler: Any, template: Any, *, max_turns: Optional[int], gym_env: Optional[str]):
        if not isinstance(scheduler, str):
            return scheduler
        from swift.rollout.multi_turn import multi_turns
        if scheduler not in multi_turns:
            raise ValueError(f'Unknown multi_turn_scheduler {scheduler!r}; available: {sorted(multi_turns)}.')
        kwargs = {'max_turns': max_turns, 'tokenizer': getattr(template, 'tokenizer', None)}
        if gym_env is not None:
            kwargs['gym_env'] = gym_env
        return multi_turns[scheduler](**kwargs)

    def generate(self,
                 prompts: List[List[dict]],
                 num_samples: int = 1,
                 sampling_params: Optional[dict] = None,
                 prompt_extras: Optional[List[Dict[str, Any]]] = None) -> List[RolloutSample]:
        from twinkle.data_format import SamplingParams

        if num_samples < 1:
            raise ValueError('num_samples must be >= 1.')
        if prompt_extras is not None and len(prompt_extras) != len(prompts):
            raise ValueError('prompt_extras must contain exactly one mapping per prompt.')
        extras = prompt_extras or [{} for _ in prompts]
        trajectories, metadata = [], []
        for prompt_index, prompt in enumerate(prompts):
            for _ in range(num_samples):
                extra = copy.deepcopy(extras[prompt_index])
                trajectories.append({'messages': copy.deepcopy(prompt), '_swift_extra': extra})
                metadata.append((str(prompt_index), extra))

        params = dict(sampling_params or {})
        params.setdefault('temperature', 1.0)
        params.setdefault('max_tokens', 32)
        params['logprobs'] = max(int(params.get('logprobs') or 0), 1)
        params['num_samples'] = 1
        outputs = self.rollout(trajectories, sampling_params=SamplingParams(**params))
        if len(outputs) != len(metadata):
            raise RuntimeError(f'multi-turn rollout returned {len(outputs)} trajectories for {len(metadata)} inputs.')
        return [
            RolloutSample(
                encoded=output['encoded'],
                response_token_ids=output['response_token_ids'],
                response_loss_mask=output['response_loss_mask'],
                rollout_logprobs=output['rollout_logprobs'],
                prompt_id=prompt_id,
                extra=extra,
                decoded=output.get('decoded', ''),
                truncated=bool(output.get('truncated')),
                messages=copy.deepcopy(output.get('messages')),
                rollout_infos=copy.deepcopy(output.get('rollout_infos') or {}),
            ) for output, (prompt_id, extra) in zip(outputs, metadata)
        ]
