import queue
import unittest
from unittest.mock import Mock, patch

from swift.infer_engine.infer_engine import InferEngine


class _TestEngine(InferEngine):

    async def infer_async(self, infer_request, request_config, **kwargs):
        raise NotImplementedError


class _TimeoutQueue(queue.Queue):

    def get(self, block=True, timeout=None):
        return super().get(block=block, timeout=0.5)


async def _failing_stream():

    async def _generator():
        raise RuntimeError('stream failed')
        yield

    return _generator()


class TestInferEngine(unittest.TestCase):

    def setUp(self):
        self.engine = object.__new__(_TestEngine)
        self.progress = Mock()

    @patch('swift.infer_engine.infer_engine.Queue', _TimeoutQueue)
    def test_stream_error_is_raised_in_strict_mode(self):
        self.engine.strict = True
        iterator = self.engine.async_iter_to_iter(_failing_stream(), self.progress, metrics=None)

        with self.assertRaisesRegex(RuntimeError, 'stream failed'):
            next(iterator)

        self.progress.update.assert_called_once_with()

    @patch('swift.infer_engine.infer_engine.Queue', _TimeoutQueue)
    def test_stream_error_ends_iteration_in_non_strict_mode(self):
        self.engine.strict = False
        iterator = self.engine.async_iter_to_iter(_failing_stream(), self.progress, metrics=None)

        with self.assertRaises(StopIteration):
            next(iterator)

        self.progress.update.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
