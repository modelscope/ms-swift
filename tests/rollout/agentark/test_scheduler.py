from __future__ import annotations

import asyncio
import unittest

from ._fakes import IMAGE_URL, FakeAgentArkClient, append_generated_assistant, delta_messages, make_choice, make_request


def _step_payload(assistant: str, *, reward: float = 0.25, done: bool = False):
    return {
        'unity_id': 0,
        'obs': {
            'messages': delta_messages(assistant)
        },
        'reward': reward,
        'done': done,
        'info': {
            'server_detail': 'kept'
        },
    }


class AgentArkSchedulerTests(unittest.IsolatedAsyncioTestCase):

    async def _started_scheduler(
        self,
        *,
        loss_scope: str = 'all_turns',
        step_payloads=None,
        max_turns: int | None = 4,
    ):
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.multi_turn import AgentArkScheduler

        client = FakeAgentArkClient(step_payloads=step_payloads)
        request = make_request(uuid='trajectory-1', group_uid='group-1', loss_scope=loss_scope)
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        scheduler = AgentArkScheduler(max_turns=max_turns)
        scheduler._create_env = lambda _env_config: env
        await scheduler.on_trajectory_start([request])
        return scheduler, request, env, client

    async def test_start_replaces_ticket_with_complete_messages(self):
        scheduler, request, _env, _client = await self._started_scheduler()

        self.assertEqual([message['role'] for message in request.messages], ['system', 'user'])
        self.assertEqual(request.images, [])
        self.assertEqual(request.messages[1]['content'][1]['image_url']['url'], IMAGE_URL)
        await scheduler.finalize_trajectory(request.uuid, reason='test_cleanup')

    async def test_step_deduplicates_assistant_and_keeps_inline_image(self):
        assistant = 'ExecutePlan R1'
        scheduler, request, _env, client = await self._started_scheduler(step_payloads=[_step_payload(assistant)])
        choice = make_choice(assistant, token_ids=[31, 32, 33])
        append_generated_assistant(request, assistant)

        turn_result = await scheduler.on_turn_end(request, choice, current_turn=1)
        step_result = scheduler.step(request, choice, current_turn=1)

        self.assertFalse(turn_result['done'])
        self.assertEqual([message['role'] for message in request.messages], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(
            [message['content'] for message in request.messages if message['role'] == 'assistant'],
            [assistant],
        )
        self.assertEqual(request.messages[-1]['content'][1]['image_url']['url'], IMAGE_URL)
        self.assertEqual(step_result['response_token_ids'], [31, 32, 33])
        self.assertEqual(step_result['response_loss_mask'], [1, 1, 1])
        self.assertEqual(client.step_calls[0]['assistant'], assistant)
        self.assertEqual(client.step_calls[0]['action_id'], 'trajectory-1:1')
        self.assertEqual(client.step_calls[0]['turn_index'], 1)
        await scheduler.finalize_trajectory(request.uuid, reason='test_cleanup')

    async def test_last_round_masks_nonterminal_assistant_tokens(self):
        assistant = 'ExecutePlan L1'
        scheduler, request, _env, _client = await self._started_scheduler(
            loss_scope='last_round',
            step_payloads=[_step_payload(assistant)],
        )
        choice = make_choice(assistant, token_ids=[41, 42])
        append_generated_assistant(request, assistant)

        turn_result = await scheduler.on_turn_end(request, choice, current_turn=1)
        step_result = scheduler.step(request, choice, current_turn=1)

        self.assertFalse(turn_result['done'])
        self.assertEqual(step_result['response_token_ids'], [41, 42])
        self.assertEqual(step_result['response_loss_mask'], [0, 0])
        await scheduler.finalize_trajectory(request.uuid, reason='test_cleanup')

    async def test_done_accumulates_reward_and_releases(self):
        assistant = 'ExecutePlan U1'
        scheduler, request, _env, client = await self._started_scheduler(
            step_payloads=[_step_payload(assistant, reward=0.8, done=True)])
        choice = make_choice(assistant)
        append_generated_assistant(request, assistant)

        turn_result = await scheduler.on_turn_end(request, choice, current_turn=1)

        self.assertTrue(turn_result['done'])
        infos = turn_result['rollout_infos']
        self.assertEqual(infos['total_reward'], 0.8)
        self.assertEqual(infos['step_rewards'], [0.8])
        self.assertTrue(infos['gym_done'])
        self.assertEqual(infos['step_infos'][0]['server_detail'], 'kept')
        self.assertEqual(client.release_calls, ['env-1'])
        self.assertNotIn(request.uuid, scheduler._envs)

    async def test_release_error_is_preserved_in_terminal_rollout_info(self):
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.multi_turn import AgentArkScheduler

        class FailingReleaseClient(FakeAgentArkClient):

            async def release(self, lease, **_kwargs):
                env_id = lease.env_id if hasattr(lease, 'env_id') else str(lease)
                self.release_calls.append(env_id)
                raise RuntimeError('synthetic release failure')

        assistant = 'ExecutePlan U1'
        client = FailingReleaseClient(step_payloads=[_step_payload(assistant, done=True)])
        request = make_request(uuid='release-failure', group_uid='release-failure-group')
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        scheduler = AgentArkScheduler(max_turns=4)
        scheduler._create_env = lambda _env_config: env
        await scheduler.on_trajectory_start([request])
        choice = make_choice(assistant)
        append_generated_assistant(request, assistant)

        turn_result = await scheduler.on_turn_end(request, choice, current_turn=1)

        self.assertTrue(turn_result['done'])
        self.assertIn('synthetic release failure', turn_result['rollout_infos']['release_error'])
        self.assertEqual(client.release_calls, ['env-1'])

    async def test_length_does_not_execute_truncated_action_and_releases(self):
        assistant = 'truncated C# code ...'
        scheduler, request, _env, client = await self._started_scheduler()
        choice = make_choice(assistant, finish_reason='length')
        append_generated_assistant(request, assistant)

        turn_result = await scheduler.on_turn_end(request, choice, current_turn=1)

        self.assertTrue(turn_result['done'])
        self.assertEqual(client.step_calls, [])
        self.assertEqual(client.release_calls, ['env-1'])
        self.assertEqual(turn_result['rollout_infos']['termination_reason'], 'length')

    async def test_max_turn_executes_last_action_then_releases(self):
        assistant = 'ExecutePlan D1'
        scheduler, request, _env, client = await self._started_scheduler(
            step_payloads=[_step_payload(assistant, reward=0.4, done=False)],
            max_turns=1,
        )
        choice = make_choice(assistant)
        append_generated_assistant(request, assistant)

        turn_result = await scheduler.on_turn_end(request, choice, current_turn=1)

        self.assertTrue(turn_result['done'])
        self.assertEqual(len(client.step_calls), 1)
        self.assertEqual(client.release_calls, ['env-1'])
        self.assertEqual(turn_result['rollout_infos']['termination_reason'], 'max_turns')

    async def test_step_exception_releases_before_propagating(self):
        assistant = 'ExecutePlan R2'
        scheduler, request, _env, client = await self._started_scheduler(
            step_payloads=[RuntimeError('fake Unity failure')])
        choice = make_choice(assistant)
        append_generated_assistant(request, assistant)

        with self.assertRaisesRegex(RuntimeError, 'fake Unity failure'):
            await scheduler.on_turn_end(request, choice, current_turn=1)

        self.assertEqual(client.release_calls, ['env-1'])
        self.assertNotIn(request.uuid, scheduler._envs)

    async def test_stale_lease_invalidates_trajectory_without_killing_rollout_driver(self):
        from swift.rollout.agentark.client import AgentArkStaleLeaseError

        assistant = 'ExecutePlan stale'
        scheduler, request, _env, client = await self._started_scheduler(step_payloads=[
            AgentArkStaleLeaseError(
                'lease identity does not own env_id; it may belong to an older generation',
                status_code=409,
                code='stale_lease',
            )
        ])
        choice = make_choice(assistant)
        append_generated_assistant(request, assistant)

        result = await scheduler.on_turn_end(request, choice, current_turn=1)

        self.assertTrue(result['done'])
        self.assertTrue(result['rollout_infos']['trajectory_invalid'])
        self.assertEqual(result['rollout_infos']['lease_recovery'], 'discard_trajectory')
        self.assertEqual(result['rollout_infos']['termination_reason'], 'stale_lease')
        self.assertEqual(client.release_calls, ['env-1'])
        self.assertNotIn(request.uuid, scheduler._envs)

    async def test_acquire_exception_leaves_no_scheduler_state(self):
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.multi_turn import AgentArkScheduler

        class FailingAcquireClient(FakeAgentArkClient):

            async def acquire_start(self, *args, **kwargs):
                raise RuntimeError('fake acquire failure')

        client = FailingAcquireClient()
        request = make_request(uuid='failed-trajectory', group_uid='failed-group')
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        scheduler = AgentArkScheduler(max_turns=4)
        scheduler._create_env = lambda _env_config: env

        with self.assertRaisesRegex(RuntimeError, 'fake acquire failure'):
            await scheduler.on_trajectory_start([request])

        self.assertNotIn(request.uuid, scheduler._envs)
        self.assertNotIn(request.uuid, scheduler._total_rewards)
        self.assertNotIn(request.uuid, scheduler._step_rewards)
        self.assertEqual(client.release_calls, [])

    async def test_one_failed_acquire_cleans_successful_sibling_from_same_batch(self):
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.multi_turn import AgentArkScheduler

        class FailingAcquireClient(FakeAgentArkClient):

            async def acquire_start(self, *args, **kwargs):
                raise RuntimeError('second acquire failed')

        successful_request = make_request(uuid='successful-trajectory', group_uid='successful-group')
        failed_request = make_request(uuid='failed-trajectory', group_uid='failed-group')
        successful_client = FakeAgentArkClient()
        failed_client = FailingAcquireClient()
        successful_env = AgentArkEnv(successful_request.data_dict['env_config'], client=successful_client)
        failed_env = AgentArkEnv(failed_request.data_dict['env_config'], client=failed_client)
        scheduler = AgentArkScheduler(max_turns=4)

        def create_env(env_config):
            return successful_env if env_config['group_uid'] == 'successful-group' else failed_env

        scheduler._create_env = create_env
        with self.assertRaisesRegex(RuntimeError, 'second acquire failed'):
            await scheduler.on_trajectory_start([successful_request, failed_request])

        self.assertEqual(successful_client.release_calls, ['env-1'])
        self.assertEqual(scheduler._envs, {})
        self.assertEqual(scheduler._total_rewards, {})
        self.assertEqual(scheduler._step_rewards, {})

    async def test_server_driver_runs_builtin_scheduler_and_releases_lease(self):
        from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
                                                 RequestConfig, UsageInfo)
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.multi_turn import AgentArkScheduler

        response = ChatCompletionResponse(
            model='fake-model',
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role='assistant', content='ExecutePlan U1'),
                    finish_reason='stop',
                    token_ids=[71, 72],
                )
            ],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=2, total_tokens=2),
        )

        class FakeServerEngine:
            tokenizer = None
            template = None

            async def infer_async(self, _request, _config, **_kwargs):
                return response

            async def _batch_infer_stream(self, tasks, _stream, _use_tqdm, _metrics):
                return await asyncio.gather(*tasks)

        client = FakeAgentArkClient(step_payloads=[_step_payload('ExecutePlan U1', reward=1.0, done=True)])
        request = make_request(uuid='server-trajectory', group_uid='server-group')
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        scheduler = AgentArkScheduler(infer_engine=FakeServerEngine(), max_turns=4)
        scheduler._create_env = lambda _env_config: env

        outputs = await scheduler.async_infer([request], RequestConfig(), use_tqdm=False)

        self.assertEqual(outputs[0].response_token_ids, [[71, 72]])
        self.assertEqual(outputs[0].rollout_infos['total_reward'], 1.0)
        self.assertEqual(client.release_calls, ['env-1'])
        self.assertEqual(scheduler._envs, {})


class SwiftMultiTurnDriverTests(unittest.TestCase):

    @staticmethod
    def _rollout_output(content: str, token_ids: list[int]):
        from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
                                                 RolloutOutput, UsageInfo)

        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=content),
            finish_reason='stop',
            token_ids=token_ids,
        )
        response = ChatCompletionResponse(
            model='fake-model',
            choices=[choice],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=len(token_ids), total_tokens=len(token_ids)),
        )
        return RolloutOutput(response=response)

    def _run_two_turn_trajectory(self, loss_scope: str):
        from swift.infer_engine.protocol import RequestConfig
        from swift.rollout.agent_loop import run_multi_turn
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.multi_turn import AgentArkScheduler

        class FakeTokenizer:

            def decode(self, token_ids, skip_special_tokens=False):
                return f'decoded-{list(token_ids)}'

        first_assistant = 'ExecutePlan R1'
        final_assistant = 'ExecutePlan U2'
        client = FakeAgentArkClient(step_payloads=[
            _step_payload(first_assistant, reward=0.2, done=False),
            _step_payload(final_assistant, reward=0.8, done=True),
        ])
        request = make_request(uuid=f'driver-{loss_scope}', group_uid='driver-group', loss_scope=loss_scope)
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        # ms-swift 4.6 materializes ID-backed assistant history before calling
        # scheduler hooks. Production passes the real tokenizer from the
        # trainer; the driver fixture supplies the smallest equivalent.
        scheduler = AgentArkScheduler(max_turns=4, tokenizer=FakeTokenizer())
        scheduler._create_env = lambda _env_config: env
        asyncio.run(scheduler.on_trajectory_start([request]))

        first_output = self._rollout_output(first_assistant, [11, 12])
        second_output = self._rollout_output(final_assistant, [21, 22, 23])
        rollout_calls = []

        def rollout_fn(requests, _request_config):
            rollout_calls.append(list(requests))
            return [second_output]

        outputs = run_multi_turn(
            requests=[request],
            first_turn_outputs=[first_output],
            scheduler=scheduler,
            rollout_fn=rollout_fn,
            request_config=RequestConfig(),
            max_turns=4,
        )
        return outputs[0], client, rollout_calls

    def test_all_turns_exact_ids_and_masks_through_swift_driver(self):
        output, client, rollout_calls = self._run_two_turn_trajectory('all_turns')

        self.assertEqual(output.response_token_ids, [[11, 12], [21, 22, 23]])
        self.assertEqual(output.response_loss_mask, [[1, 1], [1, 1, 1]])
        self.assertEqual(
            [message['role'] for message in output.messages],
            ['system', 'user', 'assistant', 'user', 'assistant'],
        )
        self.assertEqual(output.rollout_infos['total_reward'], 1.0)
        self.assertEqual(output.rollout_infos['num_turns'], 2)
        self.assertEqual(
            [call['action_id'] for call in client.step_calls],
            ['driver-all_turns:1', 'driver-all_turns:2'],
        )
        # The Swift multi-turn driver calls rollout_fn once with the pending second turn and
        # once more with an empty list before its distributed stop check.
        self.assertEqual(len(rollout_calls), 2)
        self.assertEqual(len(rollout_calls[0]), 1)
        self.assertEqual(rollout_calls[1], [])
        self.assertEqual(client.release_calls, ['env-1'])

    def test_last_round_masks_first_turn_but_not_final_turn_in_swift_driver(self):
        output, client, _rollout_calls = self._run_two_turn_trajectory('last_round')

        self.assertEqual(output.response_token_ids, [[11, 12], [21, 22, 23]])
        self.assertEqual(output.response_loss_mask, [[0, 0], [1, 1, 1]])
        self.assertEqual(client.release_calls, ['env-1'])


if __name__ == '__main__':
    unittest.main()
