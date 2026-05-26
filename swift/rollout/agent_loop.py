# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-neutral multi-turn rollout driver.

The body of this loop is lifted verbatim from
``swift/rlhf_trainers/rollout_mixin.py::RolloutTrainerMixin._colocate_multi_turn_infer``.
Only two pieces of behavior are parameterized so the same loop can run from
the HF Accelerate process group, the Megatron rollout group, or the
Megatron-Ray driver process:

* ``rollout_fn``  - run one turn of inference on a list of requests.
* ``gather_fn``   - gather a boolean across the relevant process group, so
                    all ranks can agree on the termination condition.
"""
import asyncio
from typing import Callable, List, Optional

from swift.infer_engine import RequestConfig
from swift.infer_engine.protocol import ChatCompletionResponseChoice, RolloutInferRequest, RolloutOutput
from swift.utils import remove_response
from .multi_turn import MultiTurnScheduler

RolloutFn = Callable[[List[RolloutInferRequest], RequestConfig], List[RolloutOutput]]
# gather_fn is expected to follow accelerate's `gather_object` convention:
# each rank passes a list (its contribution), and the return value is the
# concatenated flat list across all ranks. Passing a scalar would crash
# accelerate's implementation (it iterates each rank's contribution).
GatherFn = Callable[[List[bool]], List[bool]]


def _identity_gather(values: List[bool]) -> List[bool]:
    return list(values)


def invoke_async_hook(coro):
    """Run an async scheduler hook from synchronous context (colocate/Ray mode).

    Creates a temporary event loop to drive the coroutine. This is safe in
    colocate mode where no event loop is running. For server mode (where an
    event loop is active), hooks are awaited directly in ``run()``.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def extract_logprobs_from_choice(response_choice: ChatCompletionResponseChoice) -> List[float]:
    """Extract logprobs list from response choice for rollout importance sampling."""
    if response_choice.logprobs is None:
        return []
    if 'content' in response_choice.logprobs:
        return [item['logprob'] for item in response_choice.logprobs['content']]
    return []


def run_multi_turn(
    requests: List[RolloutInferRequest],
    first_turn_outputs: List[RolloutOutput],
    scheduler: MultiTurnScheduler,
    rollout_fn: RolloutFn,
    request_config: RequestConfig,
    max_turns: Optional[int] = None,
    gather_fn: GatherFn = _identity_gather,
) -> List[RolloutOutput]:
    """Drive a multi-turn rollout until every request has finished.

    ``requests`` and ``first_turn_outputs`` are 1:1 and ordered. The returned
    list is ordered to match ``requests``. Each ``RolloutOutput`` carries the
    per-turn ``response_token_ids`` / ``response_loss_mask`` /
    ``rollout_logprobs`` accumulated across the trajectory.

    ``rollout_fn`` is called once per turn with whatever requests are still
    pending on this rank (or an empty list when this rank has nothing left
    but others do). ``gather_fn`` synchronizes the "any rank still has work"
    flag across the distributed group; for single-process drivers the
    default identity is correct.
    """
    loop = asyncio.new_event_loop()
    try:
        return _run_multi_turn_impl(loop, requests, first_turn_outputs, scheduler, rollout_fn, request_config,
                                    max_turns, gather_fn)
    finally:
        loop.close()


def _run_multi_turn_impl(
    loop: asyncio.AbstractEventLoop,
    requests: List[RolloutInferRequest],
    first_turn_outputs: List[RolloutOutput],
    scheduler: MultiTurnScheduler,
    rollout_fn: RolloutFn,
    request_config: RequestConfig,
    max_turns: Optional[int] = None,
    gather_fn: GatherFn = _identity_gather,
) -> List[RolloutOutput]:
    orig_size = len(requests)
    rollout_outputs: List[Optional[RolloutOutput]] = [None] * orig_size
    rollout_infos: List[dict] = [{} for _ in range(orig_size)]
    response_token_ids: List[List[List[int]]] = [[] for _ in range(orig_size)]
    response_loss_mask: List[List[List[int]]] = [[] for _ in range(orig_size)]
    rollout_logprobs: List[List[List[float]]] = [[] for _ in range(orig_size)]
    is_continuations: List[bool] = [False] * orig_size

    index_to_infer = list(range(orig_size))
    current_turn = 1
    outputs = first_turn_outputs

    while True:
        has_local_data = bool(len(index_to_infer) > 0)
        has_global_data = gather_fn([has_local_data])
        if not any(has_global_data):
            break
        assert len(index_to_infer) == len(outputs)
        for index, output in zip(index_to_infer, outputs):
            messages = requests[index].messages
            if messages[-1]['content'] is None:
                # for continuation, we add dummy response, remove here
                remove_response(messages)
            response = output.response
            response_choice = response.choices[0]
            completion = response_choice.message.content
            is_continuations[index] = False
            if messages[-1]['role'] == 'assistant':
                messages[-1]['content'] += completion
                is_continuations[index] = True
            else:
                messages.append({'role': 'assistant', 'content': completion})

        current_requests = [requests[index] for index in index_to_infer]

        async def _gather_turn_ends():
            return list(await asyncio.gather(*[
                scheduler.on_turn_end(req, output.response.choices[0], current_turn)
                for req, output in zip(current_requests, outputs)
            ]))

        turn_results = loop.run_until_complete(_gather_turn_ends())
        for tr, index in zip(turn_results, index_to_infer):
            if tr.get('rollout_infos'):
                rollout_infos[index].update(tr['rollout_infos'])

        should_stops = [
            tr.get('done', scheduler.check_finished(req, output.response.choices[0], current_turn))
            for tr, req, output in zip(turn_results, current_requests, outputs)
        ]

        next_turn_index_to_infer: List[int] = []
        for stop, index, output in zip(should_stops, index_to_infer, outputs):
            if max_turns:
                stop = stop or (current_turn >= max_turns)
            if stop:
                is_continuation = is_continuations[index]
                response_choice = output.response.choices[0]
                current_logprobs = extract_logprobs_from_choice(response_choice)
                final_token_ids = response_choice.token_ids

                if is_continuation and response_token_ids[index]:
                    response_token_ids[index][-1].extend(final_token_ids)
                    if response_loss_mask[index]:
                        response_loss_mask[index][-1].extend([1] * len(final_token_ids))
                    if rollout_logprobs[index] and current_logprobs:
                        rollout_logprobs[index][-1].extend(current_logprobs)
                elif not response_token_ids[index]:
                    if final_token_ids:
                        response_token_ids[index] = [list(final_token_ids)]
                        response_loss_mask[index] = [[1] * len(final_token_ids)]
                    if current_logprobs:
                        rollout_logprobs[index] = [current_logprobs]
                else:
                    if final_token_ids:
                        response_token_ids[index].append(list(final_token_ids))
                        response_loss_mask[index].append([1] * len(final_token_ids))
                    if current_logprobs:
                        rollout_logprobs[index].append(current_logprobs)

                # Validate rollout_logprobs completeness: if logprobs are incomplete (missing for some turns),
                # clear them to disable rollout importance sampling correction (which requires complete logprobs).
                # rollout_logprobs should match the number of loss_mask=1 tokens, not total response tokens,
                # because completion_mask in grpo_trainer is based on labels != -100, which corresponds to loss_mask=1
                final_rollout_logprobs = rollout_logprobs[index]
                if rollout_logprobs[index]:
                    total_logprob_count = sum(len(turn_lps) for turn_lps in rollout_logprobs[index])
                    if response_loss_mask[index]:
                        total_loss_mask_1_count = sum(sum(mask) for mask in response_loss_mask[index])
                        if total_loss_mask_1_count != total_logprob_count:
                            final_rollout_logprobs = []
                    else:
                        if response_token_ids[index]:
                            total_token_count = sum(len(turn_ids) for turn_ids in response_token_ids[index])
                            if total_token_count != total_logprob_count:
                                final_rollout_logprobs = []
                        else:
                            final_rollout_logprobs = []

                rollout_outputs[index] = RolloutOutput(
                    response=output.response,
                    messages=requests[index].messages,
                    response_token_ids=response_token_ids[index],
                    response_loss_mask=response_loss_mask[index],
                    rollout_infos={
                        **rollout_infos[index],
                        'num_turns': current_turn,
                    },
                    rollout_logprobs=final_rollout_logprobs)
                continue

            is_continuation = is_continuations[index]
            step_result = scheduler.step(requests[index], output.response.choices[0], current_turn)
            current_request: RolloutInferRequest = step_result['infer_request']
            return_token_id = False
            if 'response_token_ids' in step_result:
                if is_continuation and response_token_ids[index]:
                    response_token_ids[index][-1].extend(step_result['response_token_ids'])
                else:
                    response_token_ids[index].append(step_result['response_token_ids'])
                return_token_id = True
            if 'response_loss_mask' in step_result:
                assert return_token_id, 'You must return response_token_ids with response_loss_mask return'
                assert len(step_result['response_loss_mask']) == len(step_result['response_token_ids']), \
                    'response_loss_mask must have the same length as response_token_ids'
                if is_continuation and response_loss_mask[index]:
                    response_loss_mask[index][-1].extend(step_result['response_loss_mask'])
                else:
                    response_loss_mask[index].append(step_result['response_loss_mask'])

            if 'rollout_infos' in step_result:
                # Always overwrite the rollout info for this step.
                # If you need to keep all step-wise details, switch to append or merge instead.
                rollout_infos[index].update(step_result['rollout_infos'])

            # Prefer step's returned logprobs (which may be modified/truncated) over raw response_choice logprobs.
            if 'rollout_logprobs' in step_result and step_result['rollout_logprobs']:
                current_logprobs = step_result['rollout_logprobs']
            else:
                current_logprobs = extract_logprobs_from_choice(output.response.choices[0])
            if current_logprobs:
                if is_continuation and rollout_logprobs[index]:
                    rollout_logprobs[index][-1].extend(current_logprobs)
                else:
                    rollout_logprobs[index].append(current_logprobs)

            requests[index] = current_request
            next_turn_index_to_infer.append(index)

        current_turn += 1
        infer_requests = [requests[index] for index in next_turn_index_to_infer]
        outputs = rollout_fn(infer_requests if has_local_data else [], request_config)
        index_to_infer = next_turn_index_to_infer

    assert all(o is not None for o in rollout_outputs)
    return rollout_outputs  # type: ignore[return-value]
