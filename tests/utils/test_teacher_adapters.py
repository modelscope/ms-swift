import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.arguments import RLHFArguments
from swift.pipelines.train.rlhf import SwiftRLHF


class TestTeacherAdapters(unittest.TestCase):

    def test_same_model_shortcut_respects_teacher_adapters(self):
        for rlhf_type in ('gkd', 'grpo'):
            for teacher_adapters in ([], ['teacher-lora']):
                with self.subTest(rlhf_type=rlhf_type, teacher_adapters=teacher_adapters):
                    args = SimpleNamespace(
                        rlhf_type=rlhf_type,
                        model='base',
                        teacher_model='base',
                        teacher_model_server=None,
                        teacher_adapters=teacher_adapters,
                        tuner_type='lora',
                        use_liger_kernel=False,
                        num_generations=2,
                    )
                    RLHFArguments._check_teacher(args)
                    self.assertEqual(args._teacher_use_disable_adapter, not teacher_adapters)
                    self.assertEqual(args.teacher_model, 'base' if teacher_adapters else None)
                    self.assertEqual(args.teacher_adapters, teacher_adapters)

    def test_independent_and_full_teacher_remain_explicit(self):
        for tuner_type, teacher_model in (('lora', 'other-base'), ('full', 'base')):
            with self.subTest(tuner_type=tuner_type, teacher_model=teacher_model):
                args = SimpleNamespace(
                    rlhf_type='gkd',
                    model='base',
                    teacher_model=teacher_model,
                    teacher_model_server=None,
                    teacher_adapters=[],
                    tuner_type=tuner_type,
                    use_liger_kernel=False,
                )
                RLHFArguments._check_teacher(args)
                self.assertFalse(args._teacher_use_disable_adapter)
                self.assertEqual(args.teacher_model, teacher_model)

    def test_model_loading_routes_adapters_by_role(self):
        cases = [('teacher', [], []), ('teacher', ['teacher-lora'], ['teacher-lora']),
                 ('ref', ['teacher-lora'], ['student-lora']), ('reward', ['teacher-lora'], ['reward-lora'])]
        for key, teacher_adapters, expected in cases:
            with self.subTest(key=key, teacher_adapters=teacher_adapters):
                model = Mock()
                model.requires_grad_.return_value = model
                args = SimpleNamespace(
                    rlhf_type='gkd',
                    teacher_model='base',
                    ref_model='base',
                    reward_model='reward-base',
                    teacher_deepspeed=None,
                    use_hf=False,
                    hub_token=None,
                    get_model_processor=Mock(return_value=(model, None)),
                    adapters=['student-lora'],
                    teacher_adapters=teacher_adapters,
                    reward_adapters=['reward-lora'],
                    sequence_parallel_size=1,
                )
                pipeline = object.__new__(SwiftRLHF)
                pipeline.args = args
                pipeline._get_model_task_type = Mock(return_value=(None, None))
                with patch('swift.pipelines.train.rlhf.safe_snapshot_download', return_value='base'), \
                        patch('swift.pipelines.train.rlhf.prepare_adapter', return_value=model) as prepare_adapter, \
                        patch('swift.pipelines.train.rlhf.HfConfigFactory.set_config_attr'):
                    result = pipeline._prepare_single_model(key, key, 'test-model-type', None)

                prepare_adapter.assert_called_once_with(args, model, expected)
                self.assertIs(result[0], model)
                model.requires_grad_.assert_called_once_with(False)
                model.eval.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
