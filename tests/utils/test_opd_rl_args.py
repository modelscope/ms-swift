import unittest

from swift.arguments.rlhf_args import RLHFArguments


def _make_args(**overrides):
    args = RLHFArguments.__new__(RLHFArguments)
    args.rlhf_type = 'grpo'
    args.teacher_model = None
    args.teacher_model_server = None
    args.use_liger_kernel = False
    args.num_generations = 8
    args.model = 'Qwen/Qwen2.5-0.5B-Instruct'
    args.tuner_type = 'full'
    args._teacher_use_disable_adapter = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestCheckTeacher(unittest.TestCase):

    def test_gkd_self_distill_liger_rejected(self):
        args = _make_args(rlhf_type='gkd', use_liger_kernel=True)
        with self.assertRaisesRegex(ValueError, 'liger'):
            RLHFArguments._check_teacher(args)

    def test_grpo_no_teacher_ng1_rejected(self):
        args = _make_args(num_generations=1)
        with self.assertRaisesRegex(ValueError, 'num_generations'):
            RLHFArguments._check_teacher(args)

    def test_grpo_opd_liger_rejected(self):
        args = _make_args(teacher_model='teacher', use_liger_kernel=True)
        with self.assertRaisesRegex(ValueError, 'liger'):
            RLHFArguments._check_teacher(args)

    def test_grpo_opd_ng1_allowed(self):
        args = _make_args(teacher_model='teacher', num_generations=1)
        RLHFArguments._check_teacher(args)

    def test_both_teacher_sources_rejected(self):
        args = _make_args(teacher_model='a', teacher_model_server='http://x')
        with self.assertRaisesRegex(ValueError, 'both'):
            RLHFArguments._check_teacher(args)

    def test_lora_same_model_sets_disable_adapter(self):
        args = _make_args(
            teacher_model='Qwen/Qwen2.5-0.5B-Instruct', model='Qwen/Qwen2.5-0.5B-Instruct', tuner_type='lora')
        RLHFArguments._check_teacher(args)
        self.assertTrue(args._teacher_use_disable_adapter)
        self.assertIsNone(args.teacher_model)


class TestGRPOConfigPostInit(unittest.TestCase):

    def test_scale_rewards_bool_mapping(self):
        from swift.rlhf_trainers.arguments import GRPOConfig
        cfg = GRPOConfig(
            output_dir='/tmp/grpo_test',
            num_generations=2,
            per_device_train_batch_size=1,
            scale_rewards=True,
            log_completions=False,
        )
        self.assertEqual(cfg.scale_rewards, 'group')

    def test_num_generations_ge_1(self):
        from swift.rlhf_trainers.arguments import GRPOConfig
        with self.assertRaises(ValueError):
            GRPOConfig(
                output_dir='/tmp/grpo_test',
                num_generations=0,
                per_device_train_batch_size=1,
            )


class TestRayGRPOTrainer(unittest.TestCase):

    def test_teacher_model_server_not_implemented(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from swift.ray.megatron.grpo_trainer import GRPOTrainer

        args = SimpleNamespace(
            teacher_model=None,
            teacher_model_server='http://localhost:8000',
            teacher_model_dir=None,
            _teacher_use_disable_adapter=False,
            teacher_kl_coef=1.0,
            num_generations=8,
            advantage_estimator='grpo',
            scale_rewards='group',
            kl_in_reward=False,
            reward_funcs=[],
            use_gym_env=False,
            multi_turn_scheduler=None,
            max_turns=None,
            gym_env=None,
            router_replay_mode='disabled',
            dynamic_sample=False,
            max_resample_times=3,
            truncation_strategy='left',
            reward_weights=None,
            global_batch_size=4,
            temperature=0.9,
            beta=0.04,
            generation_batch_size=None,
            steps_per_generation=1,
        )
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer._data_info = {'_driver_args': args, 'template': MagicMock()}
        with self.assertRaises(NotImplementedError):
            GRPOTrainer._prepare_state(trainer)


if __name__ == '__main__':
    unittest.main()
