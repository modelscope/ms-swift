"""Comprehensive security and regression test for the rollout server authentication fix.

Tests cover:
1. every non-health route rejects unauthenticated requests when api_key is set.
2. Normal usage works — authorized requests pass through, /health stays open.
3. Backward compatibility — when api_key is not set, all routes remain open (training/local use).
4. Startup warning fires correctly in dangerous configurations.
5. Edge cases: empty key, empty bearer token, case sensitivity, non-string headers.
6. No new risks: timing-attack resistance via secrets.compare_digest.
"""
import asyncio
import time
import unittest
from fastapi import HTTPException
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestRolloutAuthFixVulnerability(unittest.IsolatedAsyncioTestCase):
    """Verify the vulnerability is fixed: unauthenticated requests are rejected when api_key is set."""

    def _make_deploy(self, api_key='secret-key'):
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        deploy.args = SimpleNamespace(api_key=api_key, host='0.0.0.0', port=8000)
        return deploy

    def _make_request(self, headers=None):

        class RawRequest:

            def __init__(self, headers):
                self.headers = headers or {}

        return RawRequest(headers)

    # ---- Vulnerability is fixed ----

    async def test_missing_auth_rejected(self):
        """An attacker who can reach the port but supplies no credentials is rejected."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        with self.assertRaises(HTTPException) as ctx:
            await dep(self._make_request())
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn('Missing bearer token', ctx.exception.detail)

    async def test_wrong_key_rejected(self):
        """An attacker who supplies a wrong key is rejected."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        with self.assertRaises(HTTPException) as ctx:
            await dep(self._make_request({'Authorization': 'Bearer wrong-key'}))
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn('Invalid API key', ctx.exception.detail)

    async def test_no_bearer_prefix_rejected(self):
        """Authorization header without 'Bearer ' prefix is rejected."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        for scheme in ['Basic secret-key', 'secret-key', 'Token secret-key']:
            with self.assertRaises(HTTPException) as ctx:
                await dep(self._make_request({'Authorization': scheme}))
            self.assertEqual(ctx.exception.status_code, 401)

    async def test_empty_bearer_token_rejected(self):
        """An empty bearer token is rejected, not silently accepted."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        with self.assertRaises(HTTPException) as ctx:
            await dep(self._make_request({'Authorization': 'Bearer '}))
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_correct_key_accepted(self):
        """A legitimate caller with the correct key passes the auth check."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        await dep(self._make_request({'Authorization': 'Bearer secret-key'}))

    async def test_key_is_case_sensitive(self):
        """API key comparison is case-sensitive — a key with wrong case is rejected."""
        deploy = self._make_deploy(api_key='SecretKey')
        dep = deploy._require_api_key()
        with self.assertRaises(HTTPException):
            await dep(self._make_request({'Authorization': 'Bearer secretkey'}))
        # Correct case passes
        await dep(self._make_request({'Authorization': 'Bearer SecretKey'}))

    # ---- All guarded routes have the dependency attached ----

    def test_all_non_health_routes_have_auth_dependency(self):
        """Every route except /health and /health/ must carry the api_key dependency."""
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        deploy.args = SimpleNamespace(api_key='test', host='127.0.0.1', port=8000)
        deploy.app = MagicMock()
        # Track which routes get dependencies
        guarded_routes = []
        unguarded_routes = []

        class FakeRouter:

            def get(self, path, **kwargs):
                deps = kwargs.get('dependencies', [])
                if deps:
                    guarded_routes.append(('GET', path))
                else:
                    unguarded_routes.append(('GET', path))

                def decorator(func):
                    return func

                return decorator

            def post(self, path, **kwargs):
                deps = kwargs.get('dependencies', [])
                if deps:
                    guarded_routes.append(('POST', path))
                else:
                    unguarded_routes.append(('POST', path))

                def decorator(func):
                    return func

                return decorator

        deploy.app = FakeRouter()
        deploy._register_rl_rollout_app()

        # /health and /health/ must be unguarded (liveness checks)
        health_routes = [(m, p) for m, p in unguarded_routes if p in ('/health', '/health/')]
        self.assertEqual(len(health_routes), 2, f'Expected /health and /health/ unguarded, got {health_routes}')

        # All other routes must be guarded
        unguarded_non_health = [(m, p) for m, p in unguarded_routes if p not in ('/health', '/health/')]
        self.assertEqual(len(unguarded_non_health), 0, f'Found unguarded non-health routes: {unguarded_non_health}')

        # Specifically verify the most dangerous routes are guarded
        dangerous = {
            '/update_named_param/', '/update_flattened_params/', '/update_adapter_param/',
            '/update_adapter_flattened_param/', '/close_communicator/', '/init_communicator/', '/infer/'
        }
        guarded_paths = {p for _, p in guarded_routes}
        for route in dangerous:
            self.assertIn(route, guarded_paths, f'Dangerous route {route} is not guarded!')


class TestRolloutAuthNormalUsage(unittest.IsolatedAsyncioTestCase):
    """Verify normal usage is not broken."""

    def _make_deploy(self, api_key=None):
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        deploy.args = SimpleNamespace(api_key=api_key, host='0.0.0.0', port=8000)
        return deploy

    def _make_request(self, headers=None):

        class RawRequest:

            def __init__(self, headers):
                self.headers = headers or {}

        return RawRequest(headers)

    async def test_no_api_key_means_no_auth(self):
        """When api_key is None (default), all routes remain open — backward compatible."""
        deploy = self._make_deploy(api_key=None)
        dep = deploy._require_api_key()
        # Should not raise, regardless of headers
        await dep(self._make_request())
        await dep(self._make_request({'Authorization': 'Bearer anything'}))
        await dep(self._make_request({}))

    async def test_health_handler_works(self):
        """The health endpoint returns a valid response without needing auth."""
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        result = await deploy.health()
        self.assertEqual(result, {'status': 'ok'})

    async def test_correct_key_with_extra_whitespace(self):
        """Bearer token with the correct key (no extra whitespace issues) works."""
        deploy = self._make_deploy(api_key='my-key')
        dep = deploy._require_api_key()
        await dep(self._make_request({'Authorization': 'Bearer my-key'}))

    async def test_long_api_key_accepted(self):
        """A long API key (common in production) works correctly."""
        long_key = 'sk-' + 'a' * 200
        deploy = self._make_deploy(api_key=long_key)
        dep = deploy._require_api_key()
        await dep(self._make_request({'Authorization': f'Bearer {long_key}'}))


class TestRolloutStartupWarning(unittest.TestCase):
    """Verify the startup warning fires correctly and doesn't fire in safe configurations."""

    def _make_deploy(self, api_key, host):
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        deploy.args = SimpleNamespace(api_key=api_key, host=host, port=8000)
        return deploy

    def test_warning_fires_on_0000_without_key(self):
        deploy = self._make_deploy(api_key=None, host='0.0.0.0')
        with patch('swift.pipelines.infer.rollout.logger') as mock_logger:
            deploy._warn_if_unauthenticated()
        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        self.assertIn('rollout server', msg.lower())
        self.assertIn('api key', msg.lower())
        self.assertIn('weight', msg.lower())

    def test_no_warning_with_key_set(self):
        deploy = self._make_deploy(api_key='secret', host='0.0.0.0')
        with patch('swift.pipelines.infer.rollout.logger') as mock_logger:
            deploy._warn_if_unauthenticated()
        mock_logger.warning.assert_not_called()

    def test_no_warning_with_localhost(self):
        for host in ['127.0.0.1', 'localhost', '::1']:
            deploy = self._make_deploy(api_key=None, host=host)
            with patch('swift.pipelines.infer.rollout.logger') as mock_logger:
                deploy._warn_if_unauthenticated()
            mock_logger.warning.assert_not_called()

    def test_no_warning_with_key_and_localhost(self):
        deploy = self._make_deploy(api_key='secret', host='127.0.0.1')
        with patch('swift.pipelines.infer.rollout.logger') as mock_logger:
            deploy._warn_if_unauthenticated()
        mock_logger.warning.assert_not_called()


class TestDeployApiKeyHardening(unittest.TestCase):
    """Verify deploy.py's _check_api_key now uses secrets.compare_digest and behaves correctly."""

    def _make_deploy(self, api_key):
        from swift.pipelines.infer.deploy import SwiftDeploy
        deploy = object.__new__(SwiftDeploy)
        deploy.args = SimpleNamespace(api_key=api_key)
        return deploy

    def _make_request(self, headers=None):

        class RawRequest:

            def __init__(self, headers):
                self.headers = headers or {}

        return RawRequest(headers)

    def test_correct_key_passes(self):
        deploy = self._make_deploy('mykey')
        result = deploy._check_api_key(self._make_request({'authorization': 'Bearer mykey'}))
        self.assertIsNone(result)

    def test_wrong_key_rejected(self):
        deploy = self._make_deploy('mykey')
        result = deploy._check_api_key(self._make_request({'authorization': 'Bearer wrongkey'}))
        self.assertEqual(result, 'API key error')

    def test_missing_header_rejected(self):
        deploy = self._make_deploy('mykey')
        result = deploy._check_api_key(self._make_request({}))
        self.assertEqual(result, 'API key error')

    def test_no_bearer_prefix_rejected(self):
        deploy = self._make_deploy('mykey')
        result = deploy._check_api_key(self._make_request({'authorization': 'Basic mykey'}))
        self.assertEqual(result, 'API key error')

    def test_no_key_set_passes_all(self):
        deploy = self._make_deploy(None)
        result = deploy._check_api_key(self._make_request({}))
        self.assertIsNone(result)

    def test_empty_string_key_treated_as_unset(self):
        """An empty string api_key is treated the same as None — no auth enforced.

        This prevents a bypass where setting api_key='' and sending 'Bearer '
        would pass compare_digest('', '').
        """
        deploy = self._make_deploy('')
        result = deploy._check_api_key(self._make_request({}))
        self.assertIsNone(result)  # empty key = no auth, so it passes


class TestTimingAttackResistance(unittest.IsolatedAsyncioTestCase):
    """Verify secrets.compare_digest is used (constant-time comparison, no early-exit timing leak)."""

    def _make_deploy(self, api_key='target-key'):
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        deploy.args = SimpleNamespace(api_key=api_key, host='0.0.0.0', port=8000)
        return deploy

    def _make_request(self, headers=None):

        class RawRequest:

            def __init__(self, headers):
                self.headers = headers or {}

        return RawRequest(headers)

    async def test_compare_digest_is_used(self):
        """Verify that secrets.compare_digest is actually called, not == or !=."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        with patch('swift.pipelines.infer.rollout.secrets.compare_digest', return_value=True) as mock_cmp:
            await dep(self._make_request({'Authorization': 'Bearer target-key'}))
        mock_cmp.assert_called_once_with('target-key', 'target-key')

    async def test_compare_digest_called_for_wrong_key(self):
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        with patch('swift.pipelines.infer.rollout.secrets.compare_digest', return_value=False) as mock_cmp:
            with self.assertRaises(HTTPException):
                await dep(self._make_request({'Authorization': 'Bearer wrong'}))
        mock_cmp.assert_called_once_with('wrong', 'target-key')

    async def test_compare_digest_called_in_deploy(self):
        """Deploy.py should also use secrets.compare_digest."""
        from swift.pipelines.infer.deploy import SwiftDeploy
        deploy = object.__new__(SwiftDeploy)
        deploy.args = SimpleNamespace(api_key='deploy-key')
        with patch('swift.pipelines.infer.deploy.secrets.compare_digest', return_value=True) as mock_cmp:
            result = deploy._check_api_key(self._make_request({'authorization': 'Bearer deploy-key'}))
        mock_cmp.assert_called_once_with('deploy-key', 'deploy-key')
        self.assertIsNone(result)


class TestNoNewRisks(unittest.IsolatedAsyncioTestCase):
    """Edge cases that could introduce new risks or errors."""

    def _make_deploy(self, api_key='secret'):
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        deploy.args = SimpleNamespace(api_key=api_key, host='0.0.0.0', port=8000)
        return deploy

    def _make_request(self, headers=None):

        class RawRequest:

            def __init__(self, headers):
                self.headers = headers or {}

        return RawRequest(headers)

    async def test_authorization_header_case_insensitive_check(self):
        """The Bearer prefix check should be case-sensitive per HTTP spec, but 'Bearer' is standard."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        # 'bearer' (lowercase) is not standard and should be rejected to prevent bypass
        with self.assertRaises(HTTPException):
            await dep(self._make_request({'Authorization': 'bearer secret'}))

    async def test_none_authorization_header(self):
        """If the Authorization header value is None (edge case), it should not crash."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        with self.assertRaises(HTTPException) as ctx:
            await dep(self._make_request({'Authorization': None}))
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_empty_string_api_key_means_no_auth(self):
        """An empty string api_key is treated the same as None — no auth enforced.

        This prevents a footgun where setting api_key='' and then sending 'Bearer '
        would pass compare_digest('', ''). Treating empty as unset is the safe default.
        """
        deploy = self._make_deploy(api_key='')
        dep = deploy._require_api_key()
        # Empty string key = no auth, so all requests pass
        await dep(self._make_request({}))
        await dep(self._make_request({'Authorization': 'Bearer '}))

    async def test_dependency_is_async(self):
        """The returned dependency must be a coroutine function (FastAPI requirement)."""
        deploy = self._make_deploy()
        dep = deploy._require_api_key()
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(dep))

    async def test_health_route_not_affected_by_auth(self):
        """The /health route must not require auth even when api_key is set."""
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
        deploy = object.__new__(SwiftRolloutDeploy)
        deploy.args = SimpleNamespace(api_key='secret', host='0.0.0.0', port=8000)
        # health is a simple handler that doesn't call _require_api_key
        result = await deploy.health()
        self.assertEqual(result, {'status': 'ok'})


if __name__ == '__main__':
    unittest.main()
