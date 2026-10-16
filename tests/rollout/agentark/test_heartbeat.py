from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch


class _HeartbeatHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    requests: list[dict] = []
    lock = threading.Lock()

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
            type(self).requests.append(payload)
        if self.path != '/v2/leases/heartbeat':
            self._send_json(404, {'detail': 'not found'})
            return
        items = []
        for lease in payload.get('leases', []):
            env_id = lease.get('env_id')
            if env_id == 'expired-env':
                items.append({
                    'env_id': env_id,
                    'ok': False,
                    'error': {
                        'code': 'lease_gone',
                        'message': 'synthetic expiry',
                        'retryable': False,
                    },
                })
            elif env_id == 'retryable-env':
                items.append({
                    'env_id': env_id,
                    'ok': False,
                    'error': {
                        'code': 'operation_in_progress',
                        'message': 'synthetic busy lease',
                        'retryable': True,
                    },
                })
            else:
                items.append({
                    **lease,
                    'ok': True,
                    'lease_ttl_s': 0.3,
                    'lease_expires_in_s': 0.3,
                    'replayed': False,
                })
        self._send_json(200, {'server_epoch': 'heartbeat-epoch', 'items': items})


def _wait_until(predicate, *, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class HeartbeatSupervisorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _HeartbeatHandler.requests = []
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), _HeartbeatHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        with _HeartbeatHandler.lock:
            _HeartbeatHandler.requests = []

    @staticmethod
    def _lease(env_id: str, generation: int = 1, *, ttl: float = 0.3):
        from swift.rollout.agentark.heartbeat import LeaseHandle

        return LeaseHandle(
            server_epoch='heartbeat-epoch',
            env_id=env_id,
            lease_id=f'secret-{env_id}-{generation}',
            lease_generation=generation,
            lease_ttl_s=ttl,
            heartbeat_interval_s=0.05 if ttl >= 0.15 else 0.02,
            client_id='heartbeat-test-client',
            acquire_request_id=f'acquire-{env_id}-{generation}',
            release_request_id=f'release-{env_id}-{generation}',
        )

    def test_one_daemon_batches_leases_by_server_url(self):
        from swift.rollout.agentark.heartbeat import HeartbeatSupervisor

        supervisor = HeartbeatSupervisor(default_timeout_s=1)
        first = self._lease('batch-env-1')
        second = self._lease('batch-env-2')
        try:
            supervisor.register(first, self.base_url)
            daemon = supervisor.thread
            supervisor.register(second, self.base_url)
            self.assertIs(supervisor.thread, daemon)
            self.assertTrue(daemon.daemon)
            self.assertTrue(
                _wait_until(lambda: any(len(request.get('leases', [])) == 2 for request in _HeartbeatHandler.requests)))
            batch = next(request for request in _HeartbeatHandler.requests if len(request.get('leases', [])) == 2)
            self.assertEqual(
                {lease['env_id']
                 for lease in batch['leases']},
                {'batch-env-1', 'batch-env-2'},
            )
            self.assertTrue(all(lease['heartbeat_id'] for lease in batch['leases']))
        finally:
            supervisor.shutdown()

    def test_nonretryable_item_marks_only_that_lease_expired(self):
        from swift.rollout.agentark.heartbeat import HeartbeatSupervisor, LeaseExpiredError

        supervisor = HeartbeatSupervisor(default_timeout_s=1)
        healthy = self._lease('healthy-env')
        expired = self._lease('expired-env')
        try:
            supervisor.register(healthy, self.base_url)
            supervisor.register(expired, self.base_url)
            self.assertTrue(_wait_until(lambda: expired.expired))
            self.assertFalse(healthy.expired)
            with self.assertRaisesRegex(LeaseExpiredError, 'lease_gone'):
                expired.assert_active()
            healthy.assert_active()
        finally:
            supervisor.shutdown()

    def test_retryable_failures_use_local_ttl_as_final_fence(self):
        from swift.rollout.agentark.heartbeat import HeartbeatSupervisor, LeaseExpiredError

        supervisor = HeartbeatSupervisor(default_timeout_s=1)
        lease = self._lease('retryable-env', ttl=0.12)
        try:
            supervisor.register(lease, self.base_url)
            self.assertTrue(_wait_until(lambda: lease.expired))
            with self.assertRaisesRegex(LeaseExpiredError, 'deadline elapsed'):
                lease.assert_active()
            self.assertGreaterEqual(len(_HeartbeatHandler.requests), 1)
        finally:
            supervisor.shutdown()

    def test_supervisor_instance_is_not_reused_after_fork(self):
        import os

        from swift.rollout.agentark.heartbeat import HeartbeatSupervisor

        supervisor = HeartbeatSupervisor(default_timeout_s=1)
        lease = self._lease('forked-env')
        with patch('swift.rollout.agentark.heartbeat.os.getpid', return_value=os.getpid() + 1000):
            with self.assertRaisesRegex(RuntimeError, 'after fork'):
                supervisor.register(lease, self.base_url)


if __name__ == '__main__':
    unittest.main()
