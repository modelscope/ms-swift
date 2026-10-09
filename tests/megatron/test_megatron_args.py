import dataclasses
import unittest


class TestMegatronArgs(unittest.TestCase):
    """Megatron import / args smoke test (GPU and NPU adapted).

    Covers: MegatronSftArguments initialization, MegatronRLHFArguments,
    MegatronArguments field validation.

    Why these tests are needed:
    - tests/megatron/test_train.py and test_lora.py have top-level functions
      that require multi-GPU and mcore models, too heavy for CI.
    - Megatron argument construction is a common entry point that should be
      validated even without a full training run.
    - On NPU, Megatron dependencies (mcore, MindSpeed) may not be installed,
      so we gracefully skip.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from swift.megatron import (MegatronArguments, MegatronExportArguments, MegatronPretrainArguments,
                                        MegatronRLHFArguments, MegatronSftArguments)
            cls._megatron_available = True
            cls.MegatronArguments = MegatronArguments
            cls.MegatronSftArguments = MegatronSftArguments
            cls.MegatronRLHFArguments = MegatronRLHFArguments
        except (ImportError, RuntimeError) as e:
            cls._megatron_available = False
            cls._skip_reason = str(e)

    def _skip_if_no_megatron(self):
        if not self._megatron_available:
            self.skipTest(f'Megatron dependencies not available: {self._skip_reason}')

    def test_megatron_import(self):
        self._skip_if_no_megatron()

    def test_megatron_sft_args_construction(self):
        self._skip_if_no_megatron()

        args = self.MegatronSftArguments(
            mcore_model='Qwen2-7B-Instruct-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20'],
            split_dataset_ratio=0.01,
            tensor_model_parallel_size=1,
            train_iters=1,
            skip_megatron_init=True,
        )
        self.assertEqual(args.train_iters, 1)
        self.assertEqual(args.tensor_model_parallel_size, 1)

    def test_megatron_rlhf_args_construction(self):
        self._skip_if_no_megatron()

        args = self.MegatronRLHFArguments(
            rlhf_type='grpo',
            mcore_model='Qwen2-7B-Instruct-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20'],
            reward_funcs=['format'],
            num_generations=2,
            max_completion_length=128,
            tensor_model_parallel_size=1,
            train_iters=1,
            skip_megatron_init=True,
        )
        self.assertEqual(args.rlhf_type, 'grpo')
        self.assertIn('format', args.reward_funcs)

    def test_megatron_base_args_fields(self):
        self._skip_if_no_megatron()

        expected_fields = [
            'tensor_model_parallel_size',
            'pipeline_model_parallel_size',
            'context_parallel_size',
            'sequence_parallel_size',
            'train_iters',
            'micro_batch_size',
            'global_batch_size',
            'lr',
            'min_lr',
            'bf16',
        ]
        from dataclasses import fields
        field_names = {f.name for f in fields(self.MegatronArguments)}
        for field_name in expected_fields:
            self.assertIn(field_name, field_names, f'MegatronArguments missing field: {field_name}')

    def _rlhf_args(self, advantage_estimator):
        """A MegatronRLHFArguments populated from field defaults.

        ``_init_grpo`` only reads and writes the argument object's own
        attributes, so seeding every dataclass field with its default exercises
        the real method without a model, a checkpoint or a tokenizer - which the
        construction-based cases in this file require.
        """
        cls = self.MegatronRLHFArguments
        args = cls.__new__(cls)
        for field in dataclasses.fields(cls):
            if field.default is not dataclasses.MISSING:
                setattr(args, field.name, field.default)
            elif field.default_factory is not dataclasses.MISSING:
                setattr(args, field.name, field.default_factory())
        args.advantage_estimator = advantage_estimator
        args._init_grpo()
        return args

    def test_advantage_estimator_tied_defaults(self):
        """kl_in_reward and scale_rewards follow advantage_estimator.

        Mirrors RLHFArguments._init_grpo. Before the Megatron path wired this
        up, both fields kept the class defaults regardless of
        advantage_estimator, so rloo and reinforce_plus_plus silently ran with
        GRPO's settings (kl_in_reward=False, scale_rewards='group').
        """
        self._skip_if_no_megatron()

        expected = {
            'grpo': (False, 'group'),
            'rloo': (True, 'none'),
            'reinforce_plus_plus': (True, 'batch'),
        }
        for estimator, (kl_in_reward, scale_rewards) in expected.items():
            with self.subTest(advantage_estimator=estimator):
                args = self._rlhf_args(estimator)
                self.assertIs(args.kl_in_reward, kl_in_reward)
                self.assertEqual(args.scale_rewards, scale_rewards)

    def test_explicit_values_override_the_tied_defaults(self):
        """An explicit value wins over the estimator's default."""
        self._skip_if_no_megatron()

        cls = self.MegatronRLHFArguments
        args = cls.__new__(cls)
        for field in dataclasses.fields(cls):
            if field.default is not dataclasses.MISSING:
                setattr(args, field.name, field.default)
            elif field.default_factory is not dataclasses.MISSING:
                setattr(args, field.name, field.default_factory())
        args.advantage_estimator = 'rloo'
        args.kl_in_reward = False
        args.scale_rewards = 'group'
        args._init_grpo()

        self.assertIs(args.kl_in_reward, False)
        self.assertEqual(args.scale_rewards, 'group')


if __name__ == '__main__':
    unittest.main()
