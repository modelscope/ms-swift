import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.pipelines.base import SwiftPipeline
from swift.pipelines.infer.infer import SwiftInfer


class TestInferPipeline(unittest.TestCase):

    def test_infer_engine_uses_data_strict_setting(self):
        args = SimpleNamespace(
            merge_lora=False,
            infer_backend='transformers',
            adapters=[],
            max_batch_size=1,
            data_seed=42,
            strict=False,
        )
        engine = SimpleNamespace(model='model')

        def init_pipeline(pipeline, pipeline_args):
            pipeline.args = pipeline_args

        with patch.object(SwiftPipeline, '__init__', init_pipeline), \
                patch('swift.pipelines.infer.infer.prepare_model_template', return_value=('model', 'template')), \
                patch('swift.pipelines.infer.infer.TransformersEngine', return_value=engine):
            pipeline = SwiftInfer(args)

        self.assertFalse(pipeline.infer_engine.strict)

    def test_batch_infer_skips_failed_samples(self):
        pipeline = object.__new__(SwiftInfer)
        pipeline.args = SimpleNamespace(
            task_type='causal_lm',
            infer_backend='transformers',
            rank=-1,
            global_world_size=1,
            vllm_tensor_parallel_size=1,
        )
        pipeline.infer_kwargs = {}

        def response(content):
            message = SimpleNamespace(content=content)
            choice = SimpleNamespace(message=message, logprobs=None)
            return SimpleNamespace(choices=[choice])

        pipeline.infer = Mock(return_value=[response('first'), ValueError('invalid media'), response('third')])
        dataset = [{
            'messages': [{
                'role': 'user',
                'content': 'one'
            }, {
                'role': 'assistant',
                'content': 'label one'
            }]
        }, {
            'messages': [{
                'role': 'user',
                'content': 'two'
            }, {
                'role': 'assistant',
                'content': 'label two'
            }]
        }, {
            'messages': [{
                'role': 'user',
                'content': 'three'
            }, {
                'role': 'assistant',
                'content': 'label three'
            }]
        }]

        result = pipeline._batch_infer(dataset, request_config=object())

        self.assertEqual([item['response'] for item in result], ['first', 'third'])
        self.assertEqual([item['labels'] for item in result], ['label one', 'label three'])


if __name__ == '__main__':
    unittest.main()
