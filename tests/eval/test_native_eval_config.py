# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import tempfile
import unittest
from evalscope.constants import EvalBackend
from evalscope.run import TaskConfig
from pydantic import SecretStr
from types import SimpleNamespace
from unittest import mock

from swift.pipelines.eval.eval import SwiftEval


class TestNativeEvalConfig(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.pipeline = SwiftEval.__new__(SwiftEval)
        self.pipeline.args = SimpleNamespace(
            eval_output_dir=directory.name,
            model_suffix='test-model',
            api_key=None,
            eval_limit=3,
            eval_num_proc=2,
            eval_dataset_args={'gsm8k': {
                'few_shot_num': 0
            }},
            eval_generation_config={
                'max_tokens': 16,
                'temperature': 0
            },
            extra_eval_args={'ignore_errors': False})

    def test_native_backend_builds_openai_api_config(self):
        url = 'http://127.0.0.1:8000/v1'
        config = self.pipeline.get_task_cfg(['gsm8k'], EvalBackend.NATIVE, url)

        self.assertIsInstance(config, TaskConfig)
        self.assertEqual(config.eval_type, 'openai_api')
        self.assertEqual(config.model, 'test-model')
        self.assertEqual(config.api_url, url)
        api_key = config.api_key
        self.assertEqual(api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key, 'EMPTY')
        self.assertEqual(config.datasets, ['gsm8k'])
        self.assertEqual(config.work_dir, os.path.join(self.pipeline.args.eval_output_dir, 'native'))
        self.assertEqual(config.limit, 3)
        self.assertEqual(config.eval_batch_size, 2)
        self.assertEqual(config.dataset_args, self.pipeline.args.eval_dataset_args)
        self.assertFalse(config.ignore_errors)

    def test_native_backend_preserves_api_key(self):
        self.pipeline.args.api_key = 'test-api-key'
        config = self.pipeline.get_native_task_cfg(['gsm8k'], 'http://127.0.0.1:8000/v1')
        api_key = config.api_key
        self.assertEqual(api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key, 'test-api-key')

    def test_native_backend_supports_legacy_service_constant(self):
        legacy_eval_type = SimpleNamespace(SERVICE='openai_api')
        with mock.patch('swift.pipelines.eval.eval.EvalType', legacy_eval_type):
            config = self.pipeline.get_native_task_cfg(['gsm8k'], 'http://127.0.0.1:8000/v1')
        self.assertEqual(config.eval_type, legacy_eval_type.SERVICE)


if __name__ == '__main__':
    unittest.main()
