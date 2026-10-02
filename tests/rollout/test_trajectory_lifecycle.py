import asyncio
import unittest

from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
                                         RequestConfig, RolloutInferRequest, UsageInfo)
from swift.rollout.agent_loop import multi_turn_lifecycle
from swift.rollout.multi_turn import GYMScheduler, MultiTurnScheduler


def make_request(uuid=None):
    return RolloutInferRequest(messages=[{'role': 'user', 'content': 'question'}], uuid=uuid)


def make_response():
    choice = ChatCompletionResponseChoice(0, ChatMessage('assistant', 'answer'), 'stop', token_ids=[11, 12])
    return ChatCompletionResponse('fake-model', [choice], UsageInfo(0, 2, 2))


class LifecycleScheduler(MultiTurnScheduler):

    def __init__(self, *args, fail_start=False, fail_end=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_start = fail_start
        self.fail_end = fail_end
        self.events = []

    async def on_trajectory_start(self, requests):
        self.events.append(('start', list(requests)))
        if self.fail_start:
            raise ValueError('start failed')

    async def on_trajectory_end(self, requests, error=None):
        self.events.append(('end', list(requests), error))
        if self.fail_end:
            raise RuntimeError('end failed')

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        return {'done': True}

    def step(self, infer_request, response_choice, current_turn):
        return {'infer_request': infer_request}


class FakeEngine:

    tokenizer = None
    template = None

    def __init__(self, error=None):
        self.error = error

    async def infer_async(self, infer_request, request_config, **kwargs):
        if self.error is not None:
            raise self.error
        return make_response()

    async def _batch_infer_stream(self, tasks, stream, use_tqdm, metrics):
        return [await task for task in tasks]


class FakeEnv:

    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class ColocateLifecycleTest(unittest.TestCase):

    def test_success_runs_start_and_end_once(self):
        scheduler = LifecycleScheduler()
        request = make_request()

        with multi_turn_lifecycle(scheduler, [request]):
            scheduler.events.append(('body', ))

        self.assertEqual([event[0] for event in scheduler.events], ['start', 'body', 'end'])
        self.assertIsNone(scheduler.events[-1][2])

    def test_initialization_failure_still_runs_end(self):
        scheduler = LifecycleScheduler(fail_start=True)
        request = make_request()

        with self.assertRaisesRegex(ValueError, 'start failed'):
            with multi_turn_lifecycle(scheduler, [request]):
                self.fail('body must not run')

        self.assertEqual([event[0] for event in scheduler.events], ['start', 'end'])
        self.assertIsInstance(scheduler.events[-1][2], ValueError)

    def test_finalization_cannot_mask_rollout_error(self):
        scheduler = LifecycleScheduler(fail_end=True)
        request = make_request()
        original = ValueError('rollout failed')

        with self.assertRaisesRegex(ValueError, 'rollout failed') as raised:
            with multi_turn_lifecycle(scheduler, [request]):
                raise original

        self.assertIs(raised.exception, original)
        self.assertIs(scheduler.events[-1][2], original)

    def test_finalization_failure_after_success_is_raised(self):
        scheduler = LifecycleScheduler(fail_end=True)

        with self.assertRaisesRegex(RuntimeError, 'end failed'):
            with multi_turn_lifecycle(scheduler, [make_request()]):
                pass


class ServerLifecycleTest(unittest.IsolatedAsyncioTestCase):

    async def test_gym_scheduler_closes_remaining_environment(self):
        scheduler = GYMScheduler()
        request = make_request('request-1')
        env = FakeEnv()
        scheduler._envs[request.uuid] = env
        scheduler._total_rewards[request.uuid] = 0.0
        scheduler._step_rewards[request.uuid] = []
        scheduler._pending_obs[request.uuid] = None

        await scheduler.on_trajectory_end([request])

        self.assertEqual(env.close_calls, 1)
        self.assertNotIn(request.uuid, scheduler._envs)

    async def test_server_scheduler_runs_end_after_success(self):
        scheduler = LifecycleScheduler(infer_engine=FakeEngine(), max_turns=1)
        request = make_request()

        outputs = await scheduler.async_infer([request], RequestConfig(n=1), use_tqdm=False)

        self.assertEqual(len(outputs), 1)
        self.assertEqual([event[0] for event in scheduler.events], ['start', 'end'])
        self.assertIsNone(scheduler.events[-1][2])

    async def test_server_scheduler_runs_end_after_generation_failure(self):
        original = ValueError('generation failed')
        scheduler = LifecycleScheduler(infer_engine=FakeEngine(original), max_turns=1)
        request = make_request()

        with self.assertRaisesRegex(ValueError, 'generation failed') as raised:
            await scheduler.async_infer([request], RequestConfig(n=1), use_tqdm=False)

        self.assertIs(raised.exception, original)
        self.assertEqual([event[0] for event in scheduler.events], ['start', 'end'])
        self.assertIs(scheduler.events[-1][2], original)


if __name__ == '__main__':
    unittest.main()
