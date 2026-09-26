from __future__ import annotations

import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ._fakes import IMAGE_URL, FakeAgentArkClient, delta_messages, initial_messages, make_request


class AgentArkEnvTests(unittest.IsolatedAsyncioTestCase):

    async def test_reset_keeps_complete_multimodal_messages(self):
        from swift.rollout.agentark.env import AgentArkEnv

        client = FakeAgentArkClient()
        request = make_request(uuid='trajectory-1', group_uid='group-1')
        env_config = request.data_dict['env_config']
        env = AgentArkEnv(env_config, client=client)

        observation, info, system_message = await env.reset(request)

        self.assertEqual(observation, '')
        self.assertEqual(system_message, '')
        self.assertEqual(info['task_name'], 'task-for:group-1')
        self.assertEqual(env.env_id, 'env-1')
        self.assertEqual(env.initial_messages, initial_messages('initial:group-1'))
        image_part = env.initial_messages[1]['content'][1]
        self.assertEqual(image_part, {'type': 'image_url', 'image_url': {'url': IMAGE_URL}})
        self.assertEqual(client.acquire_calls[0]['uid'], 'group-1')
        self.assertEqual(client.acquire_calls[0]['env_cfg'], {'runtime': 'fake'})
        self.assertEqual(client.acquire_calls[0]['client_id'], 'fake-swift-client')
        self.assertTrue(client.acquire_calls[0]['acquire_request_id'].startswith('acquire-'))
        self.assertEqual(env.lease.server_epoch, 'fake-server-epoch')
        self.assertEqual(len(client.heartbeat_supervisor.register_calls), 1)

    async def test_two_rollouts_share_group_but_lease_distinct_envs(self):
        from swift.rollout.agentark.env import AgentArkEnv

        client = FakeAgentArkClient()
        first_request = make_request(uuid='trajectory-a', group_uid='shared-group')
        second_request = make_request(uuid='trajectory-b', group_uid='shared-group')
        first = AgentArkEnv(first_request.data_dict['env_config'], client=client)
        second = AgentArkEnv(second_request.data_dict['env_config'], client=client)

        first_reset, second_reset = await asyncio.gather(
            first.reset(first_request),
            second.reset(second_request),
        )

        self.assertNotEqual(first.env_id, second.env_id)
        self.assertEqual([call['uid'] for call in client.acquire_calls], ['shared-group', 'shared-group'])
        self.assertEqual(first_reset[1]['task_name'], second_reset[1]['task_name'])
        self.assertEqual(first_reset[1]['rollout_group_seed'], second_reset[1]['rollout_group_seed'])
        self.assertEqual(first.initial_messages, second.initial_messages)
        self.assertNotEqual(first.acquire_request_id, second.acquire_request_id)

        # Reconstructing the same trajectory produces the same acquire id;
        # group siblings never share it.
        replay = AgentArkEnv(first_request.data_dict['env_config'], client=client)
        await replay.reset(first_request)
        self.assertEqual(first.acquire_request_id, replay.acquire_request_id)
        await asyncio.gather(first.close(), second.close(), replay.close())

    async def test_step_preserves_delta_reward_done_and_info(self):
        from swift.rollout.agentark.env import AgentArkEnv

        assistant = '<tool_call>{"name":"ExecutePlan","arguments":{"plan":"R1"}}</tool_call>'
        payload = {
            'unity_id': 0,
            'obs': {
                'messages': delta_messages(assistant, 'terminal frame')
            },
            'reward': 0.75,
            'done': True,
            'info': {
                'status': 'success',
                'attempt': 2
            },
        }
        client = FakeAgentArkClient(step_payloads=[payload])
        request = make_request(uuid='trajectory-1', group_uid='group-1')
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        await env.reset(request)

        observation, reward, done, info = await env.step(
            [*copy.deepcopy(env.initial_messages), {
                'role': 'assistant',
                'content': assistant
            }],
            action_id='trajectory-1:1',
            turn_index=1,
        )

        self.assertEqual(observation, '')
        self.assertEqual(reward, 0.75)
        self.assertTrue(done)
        self.assertEqual(info['status'], 'success')
        self.assertEqual(env.pending_messages, payload['obs']['messages'])
        self.assertEqual(client.step_calls[0]['assistant'], assistant)
        self.assertEqual(client.step_calls[0]['action_id'], 'trajectory-1:1')
        self.assertEqual(client.step_calls[0]['turn_index'], 1)

    async def test_close_releases_only_once(self):
        from swift.rollout.agentark.env import AgentArkEnv

        client = FakeAgentArkClient()
        request = make_request(uuid='trajectory-1', group_uid='group-1')
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        await env.reset(request)

        await env.close()
        await env.close()

        self.assertEqual(client.release_calls, ['env-1'])
        self.assertEqual(len(client.heartbeat_supervisor.unregister_calls), 1)
        self.assertEqual(
            client.release_payloads[0]['release_request_id'],
            env.release_request_id,
        )

    async def test_expired_heartbeat_fences_next_step_locally(self):
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.agentark.heartbeat import LeaseExpiredError

        client = FakeAgentArkClient()
        request = make_request(uuid='trajectory-expired', group_uid='group-expired')
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        await env.reset(request)
        env.lease.mark_expired('synthetic stale generation')

        with self.assertRaisesRegex(LeaseExpiredError, 'synthetic stale generation'):
            await env.step(
                [*copy.deepcopy(env.initial_messages), {
                    'role': 'assistant',
                    'content': 'R1'
                }],
                action_id='trajectory-expired:1',
                turn_index=1,
            )

        self.assertEqual(client.step_calls, [])
        await env.close()

    async def test_protocol_v1_compatibility_uses_no_lease_or_heartbeat(self):
        from swift.rollout.agentark.env import AgentArkEnv

        client = FakeAgentArkClient(protocol_version='v1')
        request = make_request(uuid='legacy-trajectory', group_uid='legacy-group')
        request.data_dict['env_config']['protocol_version'] = 'v1'
        env = AgentArkEnv(request.data_dict['env_config'], client=client)
        await env.reset(request)

        self.assertIsNone(env.lease)
        self.assertIsNone(env.acquire_request_id)
        self.assertEqual(client.heartbeat_supervisor.register_calls, [])
        await env.step([*copy.deepcopy(env.initial_messages), {'role': 'assistant', 'content': 'legacy R1'}])
        await env.close()

        self.assertIsNone(client.step_calls[0]['action_id'])
        self.assertEqual(client.release_calls, ['env-1'])


class RuntimeConfigTests(unittest.TestCase):

    def test_full_runtime_yaml_selects_env_cfg_expands_values_and_deep_merges(self):
        from swift.rollout.agentark.env import resolve_runtime_config

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path = Path(temp_dir) / 'agentark-runtime.yaml'
            runtime_path.write_text(
                """
server:
  host: 127.0.0.1
  port: 18080
env_cfg:
  mod_path: ${AGENTARK_TEST_ROOT}/mods/base
  nested:
    keep: from-file
    override: from-file
    home_path: ~/agentark-assets
  paths:
    - ${AGENTARK_TEST_ROOT}/one
interaction:
  enabled: true
""".lstrip(),
                encoding='utf-8',
            )
            env_config = {
                'runtime_config_path': str(runtime_path),
                'agentark_env_cfg': {
                    'nested': {
                        'override': 'from-inline',
                        'inline_path': '${AGENTARK_TEST_ROOT}/inline',
                    },
                    'paths': ['${AGENTARK_TEST_ROOT}/replacement'],
                },
            }
            with patch.dict(os.environ, {'AGENTARK_TEST_ROOT': '/tmp/agentark-test-root'}):
                resolved = resolve_runtime_config(env_config)

        self.assertNotIn('server', resolved)
        self.assertNotIn('interaction', resolved)
        self.assertEqual(resolved['mod_path'], '/tmp/agentark-test-root/mods/base')
        self.assertEqual(resolved['nested']['keep'], 'from-file')
        self.assertEqual(resolved['nested']['override'], 'from-inline')
        self.assertEqual(Path(resolved['nested']['home_path']), Path.home() / 'agentark-assets')
        self.assertEqual(resolved['nested']['inline_path'], '/tmp/agentark-test-root/inline')
        self.assertEqual(resolved['paths'], ['/tmp/agentark-test-root/replacement'])


if __name__ == '__main__':
    unittest.main()
