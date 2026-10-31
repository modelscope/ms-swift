from __future__ import annotations

import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ._fakes import append_generated_assistant, delta_messages, initial_messages, make_choice, make_request


class _AgentArkHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    requests: list[tuple[str, dict]] = []
    lock = threading.Lock()
    next_env = 1
    v2_action_attempts: dict[str, int] = {}

    def log_message(self, _format, *args):
        return

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(length) or b'{}')
        with type(self).lock:
            type(self).requests.append((self.path, payload))

        if self.path == '/v1/envs/acquire_start':
            with type(self).lock:
                env_id = f'http-env-{type(self).next_env}'
                type(self).next_env += 1
            self._send_json(
                200,
                {
                    'env_id': env_id,
                    'unity_id': 0,
                    'obs': {
                        'messages': initial_messages('from-http')
                    },
                    'info': {
                        'task_name': 'http-task',
                        'rollout_group_seed': 99
                    },
                },
            )
            return

        if self.path == '/v2/envs/acquire_start':
            with type(self).lock:
                env_id = f'v2-http-env-{type(self).next_env}'
                type(self).next_env += 1
            self._send_json(
                200,
                {
                    'server_epoch': 'http-server-epoch',
                    'env_id': env_id,
                    'lease_id': f'lease-token-{env_id}',
                    'lease_generation': 7,
                    'lease_ttl_s': 30.0,
                    'heartbeat_interval_s': 10.0,
                    'acquire_request_id': payload['acquire_request_id'],
                    'unity_id': 0,
                    'obs': {
                        'messages': initial_messages('from-v2-http')
                    },
                    'info': {
                        'task_name': 'v2-http-task'
                    },
                },
            )
            return

        if self.path == '/v1/envs/fail-env/step':
            self._send_json(503, {'detail': 'synthetic timeout'})
            return

        if self.path.startswith('/v2/envs/') and self.path.endswith('/step'):
            action_id = str(payload.get('action_id') or '')
            with type(self).lock:
                attempt = type(self).v2_action_attempts.get(action_id, 0) + 1
                type(self).v2_action_attempts[action_id] = attempt
            if action_id == 'retry-503' and attempt == 1:
                self._send_json(
                    503,
                    {'detail': {
                        'code': 'operation_timeout',
                        'message': 'lost',
                        'retryable': True
                    }},
                )
                return
            if action_id == 'retry-in-progress' and attempt == 1:
                self._send_json(
                    409,
                    {'detail': {
                        'code': 'operation_in_progress',
                        'message': 'still running',
                        'retryable': True,
                    }},
                )
                return
            if action_id == 'semantic-conflict':
                self._send_json(
                    409,
                    {'detail': {
                        'code': 'lease_conflict',
                        'message': 'wrong turn',
                        'retryable': False,
                    }},
                )
                return
            if action_id == 'stale-identity' or action_id.startswith('e2e-stale:'):
                self._send_json(
                    409,
                    {
                        'detail': ('lease identity does not own env_id '
                                   "'v2-http-env'; it may belong to an older generation")
                    },
                )
                return
            if action_id == 'semantic-gone':
                self._send_json(
                    410,
                    {'detail': {
                        'code': 'lease_gone',
                        'message': 'expired',
                        'retryable': False,
                    }},
                )
                return
            self._send_json(
                200,
                {
                    'unity_id': 0,
                    'obs': {
                        'messages': delta_messages(payload.get('assistant', ''))
                    },
                    'reward': 0.75,
                    'done': False,
                    'info': {
                        'status': 'v2-ok'
                    },
                    'replayed': attempt > 1,
                },
            )
            return

        if self.path.endswith('/step'):
            self._send_json(
                200,
                {
                    'unity_id': 0,
                    'obs': {
                        'messages': delta_messages(payload.get('assistant', ''))
                    },
                    'reward': 0.5,
                    'done': False,
                    'info': {
                        'status': 'ok'
                    },
                },
            )
            return

        if self.path.endswith('/release'):
            if payload.get('release_request_id') == 'gone-release':
                self._send_json(
                    410,
                    {
                        'detail': {
                            'code': 'lease_gone',
                            'message': 'server_epoch does not match this env-server process',
                            'retryable': False,
                        }
                    },
                )
                return
            self._send_json(200, {'ok': True, **payload})
            return

        self._send_json(404, {'detail': 'not found'})


class AgentArkHttpClientTests(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        _AgentArkHandler.requests = []
        _AgentArkHandler.next_env = 1
        _AgentArkHandler.v2_action_attempts = {}
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), _AgentArkHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    async def test_protocol_paths_and_payloads(self):
        from swift.rollout.agentark.client import AgentArkHttpClient

        client = AgentArkHttpClient(
            self.base_url,
            timeout_s=2,
            release_timeout_s=2,
            protocol_version='v1',
        )
        try:
            started = await client.acquire_start(
                {'mod_path': '/fake/mod'},
                uid='group-http',
                task_name='PinnedTask',
                group_seed=17,
            )
            stepped = await client.step(
                started['env_id'],
                action='compiled action',
                assistant='raw assistant',
            )
            released = await client.release(started['env_id'])
        finally:
            await client.aclose()

        self.assertTrue(started['env_id'].startswith('http-env-'))
        self.assertEqual(stepped['reward'], 0.5)
        self.assertEqual(released, {'ok': True})
        acquire_path, acquire_body = _AgentArkHandler.requests[-3]
        step_path, step_body = _AgentArkHandler.requests[-2]
        release_path, release_body = _AgentArkHandler.requests[-1]
        self.assertEqual(acquire_path, '/v1/envs/acquire_start')
        self.assertEqual(acquire_body['cfg'], {'mod_path': '/fake/mod'})
        self.assertEqual(acquire_body['uid'], 'group-http')
        self.assertEqual(acquire_body['task_name'], 'PinnedTask')
        self.assertEqual(acquire_body['group_seed'], 17)
        self.assertEqual(step_path, f"/v1/envs/{started['env_id']}/step")
        self.assertEqual(step_body, {'action': 'compiled action', 'assistant': 'raw assistant'})
        self.assertEqual(release_path, f"/v1/envs/{started['env_id']}/release")
        self.assertEqual(release_body, {})

    async def test_step_error_is_not_blindly_retried(self):
        from swift.rollout.agentark.client import AgentArkHttpClient, AgentArkHttpError

        before = sum(path == '/v1/envs/fail-env/step' for path, _ in _AgentArkHandler.requests)
        client = AgentArkHttpClient(
            self.base_url,
            timeout_s=2,
            release_timeout_s=2,
            protocol_version='v1',
        )
        try:
            with self.assertRaises(AgentArkHttpError):
                await client.step('fail-env', action='dangerous', assistant='dangerous')
        finally:
            await client.aclose()
        after = sum(path == '/v1/envs/fail-env/step' for path, _ in _AgentArkHandler.requests)

        self.assertEqual(after - before, 1)

    async def test_default_client_works_across_closed_event_loops(self):
        """Match Swift colocate's reset/turn/finalize temporary-loop pattern."""

        from swift.rollout.agentark.client import AgentArkHttpClient

        client = AgentArkHttpClient(
            self.base_url,
            timeout_s=2,
            release_timeout_s=2,
            protocol_version='v1',
        )

        def exercise_three_loops():
            started = asyncio.run(client.acquire_start({'mod_path': '/fake/mod'}, uid='cross-loop-group'))
            stepped = asyncio.run(
                client.step(started['env_id'], action='cross-loop-action', assistant='cross-loop-assistant'))
            released = asyncio.run(client.release(started['env_id']))
            asyncio.run(client.aclose())
            return started, stepped, released

        started, stepped, released = await asyncio.to_thread(exercise_three_loops)
        self.assertTrue(started['env_id'].startswith('http-env-'))
        self.assertEqual(stepped['reward'], 0.5)
        self.assertEqual(released, {'ok': True})

    async def test_v2_flat_identity_and_idempotency_payloads(self):
        from swift.rollout.agentark.client import AgentArkHttpClient
        from swift.rollout.agentark.heartbeat import LeaseHandle

        client = AgentArkHttpClient(
            self.base_url,
            timeout_s=2,
            release_timeout_s=2,
            protocol_version='v2',
            client_id='http-test-client',
            v2_retry_base_delay_s=0,
            v2_retry_max_delay_s=0,
        )
        acquire_id = 'acquire-http-trajectory'
        started = await client.acquire_start(
            {'mod_path': '/fake/v2-mod'},
            uid='v2-group',
            task_name='V2Task',
            group_seed=23,
            acquire_request_id=acquire_id,
        )
        lease = LeaseHandle.from_acquire_response(
            started,
            client_id=client.client_id,
            acquire_request_id=acquire_id,
            release_request_id='release-http-trajectory',
        )
        stepped = await client.step(
            lease,
            action='v2 action',
            assistant='v2 assistant',
            action_id='v2-trajectory:1',
            turn_index=1,
        )
        released = await client.release(lease)
        await client.aclose()

        self.assertEqual(stepped['reward'], 0.75)
        self.assertTrue(released['ok'])
        acquire_path, acquire_body = _AgentArkHandler.requests[-3]
        step_path, step_body = _AgentArkHandler.requests[-2]
        release_path, release_body = _AgentArkHandler.requests[-1]
        self.assertEqual(acquire_path, '/v2/envs/acquire_start')
        self.assertEqual(acquire_body['acquire_request_id'], acquire_id)
        self.assertEqual(acquire_body['client_id'], 'http-test-client')
        self.assertEqual(step_path, f'/v2/envs/{lease.env_id}/step')
        self.assertEqual(step_body['server_epoch'], lease.server_epoch)
        self.assertEqual(step_body['lease_id'], lease.lease_id)
        self.assertEqual(step_body['lease_generation'], 7)
        self.assertEqual(step_body['action_id'], 'v2-trajectory:1')
        self.assertEqual(step_body['turn_index'], 1)
        self.assertEqual(release_path, f'/v2/envs/{lease.env_id}/release')
        self.assertEqual(release_body['release_request_id'], 'release-http-trajectory')

    async def test_v2_retries_only_safe_cases_with_the_same_action_id(self):
        from swift.rollout.agentark.client import AgentArkHttpClient, AgentArkHttpError
        from swift.rollout.agentark.heartbeat import LeaseHandle

        client = AgentArkHttpClient(
            self.base_url,
            timeout_s=2,
            release_timeout_s=2,
            protocol_version='v2',
            client_id='retry-client',
            v2_max_attempts=3,
            v2_retry_base_delay_s=0,
            v2_retry_max_delay_s=0,
        )
        started = await client.acquire_start(
            {'runtime': 'retry'},
            uid='retry-group',
            acquire_request_id='retry-acquire',
        )
        lease = LeaseHandle.from_acquire_response(
            started,
            client_id=client.client_id,
            acquire_request_id='retry-acquire',
            release_request_id='retry-release',
        )

        for action_id in ('retry-503', 'retry-in-progress'):
            result = await client.step(
                lease,
                action=action_id,
                assistant=action_id,
                action_id=action_id,
                turn_index=1,
            )
            self.assertEqual(result['reward'], 0.75)
            calls = [
                body for path, body in _AgentArkHandler.requests
                if path.endswith('/step') and body.get('action_id') == action_id
            ]
            self.assertEqual(len(calls), 2)
            self.assertEqual({call['action_id'] for call in calls}, {action_id})
            self.assertEqual(calls[0], calls[1])

        for action_id, status in (('semantic-conflict', 409), ('semantic-gone', 410)):
            with self.assertRaises(AgentArkHttpError) as raised:
                await client.step(
                    lease,
                    action=action_id,
                    assistant=action_id,
                    action_id=action_id,
                    turn_index=1,
                )
            self.assertEqual(raised.exception.status_code, status)
            self.assertEqual(_AgentArkHandler.v2_action_attempts[action_id], 1)
        await client.aclose()

    async def test_stale_lease_conflict_is_classified_and_not_retried(self):
        from swift.rollout.agentark.client import AgentArkHttpClient, AgentArkStaleLeaseError
        from swift.rollout.agentark.heartbeat import LeaseHandle

        client = AgentArkHttpClient(
            self.base_url,
            timeout_s=2,
            release_timeout_s=2,
            protocol_version='v2',
            client_id='stale-client',
            v2_max_attempts=3,
            v2_retry_base_delay_s=0,
            v2_retry_max_delay_s=0,
        )
        started = await client.acquire_start(
            {'runtime': 'stale'},
            uid='stale-group',
            acquire_request_id='stale-acquire',
        )
        lease = LeaseHandle.from_acquire_response(
            started,
            client_id=client.client_id,
            acquire_request_id='stale-acquire',
            release_request_id='stale-release',
        )

        with self.assertRaises(AgentArkStaleLeaseError):
            await client.step(
                lease,
                action='stale-identity',
                assistant='stale-identity',
                action_id='stale-identity',
                turn_index=1,
            )

        calls = [
            body for path, body in _AgentArkHandler.requests
            if path.endswith('/step') and body.get('action_id') == 'stale-identity'
        ]
        self.assertEqual(len(calls), 1)
        self.assertTrue(lease.expired)
        self.assertIn('stale lease identity', lease.expired_reason)
        await client.aclose()

    async def test_lease_gone_release_is_already_cleaned_up(self):
        from swift.rollout.agentark.client import AgentArkHttpClient
        from swift.rollout.agentark.heartbeat import LeaseHandle

        client = AgentArkHttpClient(
            self.base_url,
            timeout_s=2,
            release_timeout_s=2,
            protocol_version='v2',
            client_id='gone-client',
            v2_max_attempts=3,
            v2_retry_base_delay_s=0,
            v2_retry_max_delay_s=0,
        )
        started = await client.acquire_start(
            {'runtime': 'gone'},
            uid='gone-group',
            acquire_request_id='gone-acquire',
        )
        lease = LeaseHandle.from_acquire_response(
            started,
            client_id=client.client_id,
            acquire_request_id='gone-acquire',
            release_request_id='gone-release',
        )

        result = await client.release(lease)

        self.assertEqual(result, {'ok': True, 'stale_lease': True})
        self.assertTrue(lease.expired)
        self.assertIn('stale lease during release', lease.expired_reason)
        releases = [body for path, body in _AgentArkHandler.requests if path.endswith('/release')]
        self.assertEqual(releases[-1]['release_request_id'], 'gone-release')
        await client.aclose()

    async def test_stale_lease_409_through_http_env_and_scheduler_is_contained(self):
        from swift.rollout.multi_turn import AgentArkScheduler

        request = make_request(uuid='e2e-stale', group_uid='e2e-stale-group')
        request.data_dict['env_config'].update({
            'server_url': self.base_url,
            'http_timeout_s': 2,
            'release_timeout_s': 2,
            'heartbeat_timeout_s': 2,
        })
        scheduler = AgentArkScheduler(max_turns=4)
        await scheduler.on_trajectory_start([request])
        assistant = 'stale-identity'
        choice = make_choice(assistant)
        append_generated_assistant(request, assistant)
        request_count_before = len(_AgentArkHandler.requests)

        result = await scheduler.on_turn_end(request, choice, current_turn=1)

        requests_after = _AgentArkHandler.requests[request_count_before:]
        stale_steps = [
            body for path, body in requests_after if path.endswith('/step') and body.get('action_id') == 'e2e-stale:1'
        ]
        releases = [body for path, body in requests_after if path.endswith('/release')]
        self.assertEqual(len(stale_steps), 1)
        self.assertEqual(len(releases), 1)
        self.assertTrue(result['done'])
        self.assertTrue(result['rollout_infos']['trajectory_invalid'])
        self.assertEqual(result['rollout_infos']['termination_reason'], 'stale_lease')
        self.assertEqual(result['rollout_infos']['lease_recovery'], 'discard_trajectory')
        self.assertNotIn(request.uuid, scheduler._envs)


if __name__ == '__main__':
    unittest.main()
