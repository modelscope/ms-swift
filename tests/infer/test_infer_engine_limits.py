import unittest

from swift.infer_engine.infer_engine import InferEngine
from swift.infer_engine.protocol import RequestConfig


class _TestEngine(InferEngine):

    async def infer_async(self, infer_request, request_config, **kwargs):
        raise NotImplementedError


class TestInferEngineLimits(unittest.TestCase):

    @staticmethod
    def _engine(max_model_len):
        engine = object.__new__(_TestEngine)
        engine.max_model_len = max_model_len
        engine.max_tokens_offset = 0
        return engine

    def test_rejects_prompt_longer_than_context(self):
        engine = self._engine(max_model_len=4)
        request_config = RequestConfig(max_tokens=2)

        with self.assertRaisesRegex(ValueError, r'Input length \(5\).*max_model_len \(4\)'):
            engine.set_default_max_tokens(request_config, {'input_ids': [1, 2, 3, 4, 5]})

    def test_rejects_prompt_that_leaves_no_generation_space(self):
        engine = self._engine(max_model_len=4)
        request_config = RequestConfig(max_tokens=None)

        with self.assertRaisesRegex(ValueError, r'Input length \(4\).*max_model_len \(4\)'):
            engine.set_default_max_tokens(request_config, {'input_ids': [1, 2, 3, 4]})

    def test_caps_requested_tokens_to_available_context(self):
        engine = self._engine(max_model_len=5)
        request_config = RequestConfig(max_tokens=4)

        engine.set_default_max_tokens(request_config, {'input_ids': [1, 2, 3]})

        self.assertEqual(request_config.max_tokens, 2)


if __name__ == '__main__':
    unittest.main()
